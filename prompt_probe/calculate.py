import argparse
import asyncio
import math
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment
from ruamel.yaml import YAML
from tqdm.asyncio import tqdm_asyncio

from .api_client import ApiClient
from .chart import generate_chart


@dataclass
class Criterion:
    key: str
    text: str | None = None

    @classmethod
    def from_config(cls, item: str | dict) -> "Criterion":
        if isinstance(item, str):
            return cls(key=item)
        return cls(key=item["key"], text=item.get("text"))


@dataclass
class Config:
    criteria: list[Criterion]
    system_prompt: str
    variant_a: dict
    variant_b: dict
    num_runs: int
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        defaults = data.get("defaults", {})
        return cls(
            criteria=[Criterion.from_config(c) for c in data["criteria"]],
            system_prompt=data["system_prompt"],
            variant_a={**defaults, **data["variant_a"]},
            variant_b={**defaults, **data["variant_b"]},
            num_runs=data["num_runs"],
            seed=data.get("seed", 0),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the config file")
    args = parser.parse_args()

    yaml = YAML()
    with open(args.config_file) as f:
        data = yaml.load(f)

    config = Config.from_dict(data)
    responses, api = asyncio.run(run_comparisons(config))
    api.write_trace()

    results_a = defaultdict(list)
    results_b = defaultdict(list)
    expected_keys = {c.key for c in config.criteria}
    for i in range(config.num_runs):
        for key, val in parse_response(responses[i * 2], expected_keys).items():
            results_a[key].append(val)
        for key, val in parse_response(responses[i * 2 + 1], expected_keys).items():
            results_b[key].append(val)

    diffs = compute_diffs(config.criteria, results_a, results_b)
    config_path = Path(args.config_file)
    label_a = config.variant_a.get("label", "Variant A")
    label_b = config.variant_b.get("label", "Variant B")
    print_results(diffs, label_a, label_b, api.prompt_cost, api.completion_cost)
    chart_data = _build_chart_data(diffs, label_a, label_b, config.num_runs)
    generate_chart(chart_data, config_path.with_suffix(".png"))


async def run_comparisons(config: Config) -> tuple[list[str], ApiClient]:
    params_a = _variant_params(config.variant_a)
    params_b = _variant_params(config.variant_b)

    base_rng = random.Random(config.seed)
    async with ApiClient.create() as api:
        tasks = []
        for _i in range(config.num_runs):
            rng = random.Random(base_rng.getrandbits(64))
            shuffled = list(config.criteria)
            rng.shuffle(shuffled)
            shuffled_keys = [c.key for c in shuffled]
            messages_a = build_messages(config, shuffled_keys, config.variant_a)
            messages_b = build_messages(config, shuffled_keys, config.variant_b)
            tasks.append(api.complete(messages_a, **params_a, temperature=0.0))
            tasks.append(api.complete(messages_b, **params_b, temperature=0.0))

        return await _run_with_cache_warmup(tasks), api


def build_messages(
    config: Config, shuffled_keys: list[str], variant: dict
) -> list[dict]:
    criteria_defs = "\n".join(
        f"{c.key}: {c.text}" if c.text else c.key for c in config.criteria
    )
    criteria_order = "\n".join(shuffled_keys)
    subject_text = _render_template(variant["subject_text"])
    messages: list[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": config.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": subject_text}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Criteria:\n{criteria_defs}",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Order:\n{criteria_order}"}],
        },
    ]
    return _add_cache_padding(messages)


def parse_response(response: str, expected_keys: set[str]) -> dict[str, int]:
    results = {}
    pattern = re.compile(r"^(.+?)\s*(-?\d+)$")
    for line in response.strip().split("\n"):
        if match := pattern.match(line.strip()):
            results[match.group(1).strip()] = int(match.group(2))
    missing = expected_keys - results.keys()
    if missing:
        raise ValueError(f"Missing criteria: {missing}")
    return results


def compute_diffs(
    criteria: list[Criterion], results_a: dict, results_b: dict
) -> list[tuple]:
    diffs = []
    for criterion in criteria:
        vals_a = results_a[criterion.key]
        vals_b = results_b[criterion.key]
        mean_a = statistics.mean(vals_a) if vals_a else 0
        mean_b = statistics.mean(vals_b) if vals_b else 0
        sem_a = _calc_sem(vals_a)
        sem_b = _calc_sem(vals_b)
        paired_diffs = [b - a for a, b in zip(vals_a, vals_b, strict=False)]
        mean_diff = statistics.mean(paired_diffs) if paired_diffs else 0
        sem_diff = _calc_sem(paired_diffs)
        diffs.append((criterion.key, mean_a, sem_a, mean_b, sem_b, mean_diff, sem_diff))
    diffs.sort(key=lambda x: abs(x[5]), reverse=True)

    if criteria:
        num_runs = len(results_a[criteria[0].key])
        diffs.append(_compute_overall_average(criteria, results_a, results_b, num_runs))

    return diffs


def _compute_overall_average(
    criteria: list[Criterion], results_a: dict, results_b: dict, num_runs: int
) -> tuple:
    per_run_avg_a = []
    per_run_avg_b = []
    per_run_avg_diff = []
    for i in range(num_runs):
        run_vals_a = [results_a[c.key][i] for c in criteria]
        run_vals_b = [results_b[c.key][i] for c in criteria]
        avg_a = statistics.mean(run_vals_a)
        avg_b = statistics.mean(run_vals_b)
        per_run_avg_a.append(avg_a)
        per_run_avg_b.append(avg_b)
        per_run_avg_diff.append(avg_b - avg_a)
    return (
        "(Overall Average)",
        statistics.mean(per_run_avg_a),
        _calc_sem(per_run_avg_a),
        statistics.mean(per_run_avg_b),
        _calc_sem(per_run_avg_b),
        statistics.mean(per_run_avg_diff),
        _calc_sem(per_run_avg_diff),
    )


def print_results(
    diffs: list[tuple],
    label_a: str,
    label_b: str,
    prompt_cost: float,
    completion_cost: float,
):
    print(f"{'Criterion':<25} {'Diff':<15} {label_a:<20} {label_b:<20}")
    print("-" * 80)
    for attr, mean_a, sem_a, mean_b, sem_b, diff, sem_diff in diffs:
        p = (3, 2) if attr == "(Overall Average)" else (1, 1)
        diff_str = f"{diff:+.{p[0]}f} ± {sem_diff:.{p[1] + 1}f}"
        val_a_str = f"{mean_a:.{p[1]}f} ± {sem_a:.{p[1] + 1}f}"
        val_b_str = f"{mean_b:.{p[1]}f} ± {sem_b:.{p[1] + 1}f}"
        print(f"{attr:<25} {diff_str:<15} {val_a_str:<20} {val_b_str:<20}")
    total = prompt_cost + completion_cost
    print(
        f"\nCost: ${total:.4f} "
        f"(${prompt_cost:.4f} prompt + ${completion_cost:.4f} completion)"
    )


def _build_chart_data(
    diffs: list[tuple], label_a: str, label_b: str, num_runs: int
) -> dict:
    return {
        "num_runs": num_runs,
        "variants": [label_a, label_b],
        "criteria": [
            {"name": name, "diff": {"mean": diff, "sem": sem_diff}}
            for name, _, _, _, _, diff, sem_diff in diffs
        ],
    }


def _variant_params(variant: dict) -> dict:
    exclude = {"label", "subject_text"}
    return {k: v for k, v in variant.items() if k not in exclude}


async def _run_with_cache_warmup(tasks) -> list[str]:
    """
    For each variant, runs sequentially until the first real API call (hishel cache
    miss), which warms the Anthropic cache, then runs remaining in parallel.
    """
    tasks_a = tasks[::2]
    tasks_b = tasks[1::2]

    with tqdm_asyncio(total=len(tasks)) as pbar:

        async def run_variant(variant_tasks):
            results = []
            remaining = list(variant_tasks)

            while remaining:
                content, from_cache = await remaining.pop(0)
                pbar.update(1)
                results.append(content)
                if not from_cache:
                    break

            if remaining:

                async def run(i, task):
                    content, _ = await task
                    pbar.update(1)
                    return i, content

                indexed_tasks = [run(i, t) for i, t in enumerate(remaining)]
                parallel_results = [None] * len(remaining)
                for coro in asyncio.as_completed(indexed_tasks):
                    i, content = await coro
                    parallel_results[i] = content
                results.extend(parallel_results)

            return results

        results_a, results_b = await asyncio.gather(
            run_variant(tasks_a),
            run_variant(tasks_b),
        )

    return [x for pair in zip(results_a, results_b, strict=True) for x in pair]


def _calc_sem(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals) / math.sqrt(len(vals))


def _render_template(text: str) -> str:
    env = Environment()
    env.globals["load"] = lambda path: Path(path).read_text()
    return env.from_string(text).render()


def _add_cache_padding(messages: list[dict], min_tokens: int = 6000) -> list[dict]:
    def _count_message_tokens(msg: dict) -> int:
        content = msg["content"]
        if isinstance(content, list):
            text = " ".join(part.get("text", "") for part in content)
        else:
            text = content
        return len(text) // 4

    token_count = sum(_count_message_tokens(m) for m in messages[:-1])
    tokens_needed = min_tokens - token_count

    if tokens_needed <= 0:
        return messages

    padding = f'<padding ignore="true">\n{"X" * (tokens_needed * 4)}\n</padding>'
    return [
        messages[0],
        {"role": "user", "content": padding},
        *messages[1:],
    ]


if __name__ == "__main__":
    main()
