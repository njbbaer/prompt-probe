import argparse
import asyncio
import math
import os
import random
import re
import statistics
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import backoff
import hishel
import httpx
from hishel.httpx import AsyncCacheClient
from jinja2 import Environment
from ruamel.yaml import YAML
from tqdm.asyncio import tqdm_asyncio


class ApiClient:
    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self.prompt_cost = 0.0
        self.completion_cost = 0.0

    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=3)
    async def complete(self, messages: list, **params) -> tuple[str, bool]:
        response = await self._client.post(
            f"{self._BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"provider": {"order": ["anthropic"]}, "messages": messages, **params},
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(data["error"])
        from_cache = response.extensions.get("hishel_from_cache", False)
        if not from_cache:
            self.prompt_cost += data["usage"]["cost_details"][
                "upstream_inference_prompt_cost"
            ]
            self.completion_cost += data["usage"]["cost_details"][
                "upstream_inference_completions_cost"
            ]
        return data["choices"][0]["message"]["content"], from_cache

    @classmethod
    @asynccontextmanager
    async def create(cls):
        async with _create_cached_client() as client:
            yield cls(client)


@dataclass
class Config:
    attributes: list[str]
    system_prompt: str
    variant_a: dict
    variant_b: dict
    num_runs: int
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        defaults = data.get("defaults", {})
        return cls(
            attributes=data["attributes"],
            system_prompt=data["system_prompt"],
            variant_a={**defaults, **data["variant_a"]},
            variant_b={**defaults, **data["variant_b"]},
            num_runs=data["num_runs"],
            seed=data.get("seed", 0),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the config file")
    parser.add_argument(
        "-c",
        "--canary",
        action="store_true",
        help="Run single iteration and delete output",
    )
    args = parser.parse_args()

    yaml = YAML()
    with open(args.config_file) as f:
        data = yaml.load(f)

    if args.canary:
        data["num_runs"] = 1

    config = Config.from_dict(data)
    responses, api = asyncio.run(run_comparisons(config))

    results_a = defaultdict(list)
    results_b = defaultdict(list)
    for i in range(config.num_runs):
        for attr, val in parse_response(responses[i * 2]).items():
            results_a[attr].append(val)
        for attr, val in parse_response(responses[i * 2 + 1]).items():
            results_b[attr].append(val)

    diffs = compute_diffs(config.attributes, results_a, results_b)
    config_path = Path(args.config_file)
    output_path = print_results(
        diffs,
        config.variant_a,
        config.variant_b,
        config.num_runs,
        api.prompt_cost,
        api.completion_cost,
        config_path,
    )

    if args.canary and output_path.exists():
        output_path.unlink()
        print(f"Canary mode: deleted {output_path}")


async def run_comparisons(config: Config) -> tuple[list[str], ApiClient]:
    params_a = _variant_params(config.variant_a)
    params_b = _variant_params(config.variant_b)

    async with ApiClient.create() as api:
        tasks = []
        for i in range(config.num_runs):
            rng = random.Random(config.seed + i)
            shuffled = list(config.attributes)
            rng.shuffle(shuffled)
            messages_a = build_messages(config, shuffled, config.variant_a)
            messages_b = build_messages(config, shuffled, config.variant_b)
            tasks.append(api.complete(messages_a, **params_a, temperature=0.0))
            tasks.append(api.complete(messages_b, **params_b, temperature=0.0))

        return await _gather_with_warm_cache(tasks), api


def build_messages(config: Config, attributes: list[str], variant: dict) -> list[dict]:
    attributes_list = "\n".join(f"- {attr}" for attr in attributes)
    character_description = _render_template(variant.get("character_description", ""))
    messages: list[dict] = [
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": character_description,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": f"Attributes:\n{attributes_list}"},
    ]
    return _add_cache_padding(messages)


def parse_response(response: str) -> dict[str, int]:
    results = {}
    pattern = re.compile(r"^(.+?):\s*(-?\d+)$")
    for line in response.strip().split("\n"):
        if not (match := pattern.match(line.strip())):
            raise ValueError(f"Unable to parse line: '{line}'")
        results[match.group(1).strip()] = int(match.group(2))
    return results


def compute_diffs(
    attributes: list[str], results_a: dict, results_b: dict
) -> list[tuple]:
    diffs = []
    for attr in attributes:
        vals_a = results_a[attr]
        vals_b = results_b[attr]
        mean_a = statistics.mean(vals_a) if vals_a else 0
        mean_b = statistics.mean(vals_b) if vals_b else 0
        sem_a = _calc_sem(vals_a)
        sem_b = _calc_sem(vals_b)
        paired_diffs = [b - a for a, b in zip(vals_a, vals_b, strict=False)]
        mean_diff = statistics.mean(paired_diffs) if paired_diffs else 0
        sem_diff = _calc_sem(paired_diffs)
        diffs.append((attr, mean_a, sem_a, mean_b, sem_b, mean_diff, sem_diff))
    diffs.sort(key=lambda x: abs(x[5]), reverse=True)
    return diffs


def print_results(
    diffs: list[tuple],
    variant_a: dict,
    variant_b: dict,
    num_runs: int,
    prompt_cost: float,
    completion_cost: float,
    config_path: Path,
) -> Path:
    label_a = variant_a.get("label", "Variant A")
    label_b = variant_b.get("label", "Variant B")
    print(f"{'Attribute':<25} {'Diff':<15} {label_a:<20} {label_b:<20}")
    print("-" * 80)
    for attr, mean_a, sem_a, mean_b, sem_b, diff, sem_diff in diffs:
        diff_str = f"{diff:+.1f} ± {sem_diff:.2f}"
        val_a_str = f"{mean_a:.1f} ± {sem_a:.2f}"
        val_b_str = f"{mean_b:.1f} ± {sem_b:.2f}"
        print(f"{attr:<25} {diff_str:<15} {val_a_str:<20} {val_b_str:<20}")
    total = prompt_cost + completion_cost
    print(
        f"\nCost: ${total:.4f} "
        f"(${prompt_cost:.4f} prompt + ${completion_cost:.4f} completion)"
    )
    return _save_results(
        diffs, label_a, label_b, num_runs, prompt_cost, completion_cost, config_path
    )


def _save_results(
    diffs: list[tuple],
    label_a: str,
    label_b: str,
    num_runs: int,
    prompt_cost: float,
    completion_cost: float,
    config_path: Path,
) -> Path:
    output_path = config_path.with_suffix(".results.yml")
    results = {
        "num_runs": num_runs,
        "variants": [label_a, label_b],
        "cost": {
            "prompt": prompt_cost,
            "completion": completion_cost,
            "total": prompt_cost + completion_cost,
        },
        "attributes": [
            {
                "name": attr,
                "diff": {"mean": diff, "sem": sem_diff},
                label_a: {"mean": mean_a, "sem": sem_a},
                label_b: {"mean": mean_b, "sem": sem_b},
            }
            for attr, mean_a, sem_a, mean_b, sem_b, diff, sem_diff in diffs
        ],
    }
    yaml = YAML()
    yaml.default_flow_style = False
    with open(output_path, "w") as f:
        yaml.dump(results, f)
    print(f"Results saved to {output_path}")
    return output_path


def _variant_params(variant: dict) -> dict:
    exclude = {"label", "character_description"}
    return {k: v for k, v in variant.items() if k not in exclude}


async def _gather_with_warm_cache(tasks) -> list[str]:
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
                parallel_results = await asyncio.gather(*remaining)
                for content, _ in parallel_results:
                    pbar.update(1)
                    results.append(content)

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


def _create_cached_client() -> AsyncCacheClient:
    Path(".cache").mkdir(exist_ok=True)
    storage = hishel.AsyncSqliteStorage(database_path=Path(".cache/http_cache.db"))
    policy = _BodyKeyFilterPolicy()
    return AsyncCacheClient(storage=storage, policy=policy)


class _BodyKeyFilterPolicy(hishel.FilterPolicy):
    use_body_key = True


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
