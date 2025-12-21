import asyncio
import math
import os
import random
import re
import statistics
import sys
import backoff
from collections import defaultdict
from pathlib import Path

import httpx
from jinja2 import Environment
from ruamel.yaml import YAML
from tqdm.asyncio import tqdm_asyncio


class ApiClient:
    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.prompt_cost = 0.0
        self.completion_cost = 0.0

    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=3)
    async def complete(
        self, client: httpx.AsyncClient, messages: list, **params
    ) -> str:
        response = await client.post(
            f"{self._BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"provider": {"order": ["anthropic"]}, "messages": messages, **params},
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(data["error"])
        self.prompt_cost += data["usage"]["cost_details"][
            "upstream_inference_prompt_cost"
        ]
        self.completion_cost += data["usage"]["cost_details"][
            "upstream_inference_completions_cost"
        ]
        return data["choices"][0]["message"]["content"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <config_file>")
        sys.exit(1)

    yaml = YAML()
    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    api = ApiClient()
    variant_a = config["variant_a"]
    variant_b = config["variant_b"]
    responses = asyncio.run(run_comparisons(api, config, variant_a, variant_b))

    num_runs = config.get("num_runs", 1)
    results_a = defaultdict(list)
    results_b = defaultdict(list)
    for i in range(num_runs):
        for attr, val in parse_response(responses[i * 2]).items():
            results_a[attr].append(val)
        for attr, val in parse_response(responses[i * 2 + 1]).items():
            results_b[attr].append(val)

    diffs = compute_diffs(config["attributes"], results_a, results_b)
    print_results(diffs, variant_a, variant_b, api.prompt_cost, api.completion_cost)


async def run_comparisons(
    api: ApiClient, config: dict, variant_a: dict, variant_b: dict
) -> list[str]:
    num_runs = config.get("num_runs", 1)
    warm_cache = config.get("warm_cache", False)
    params_a = _variant_params(variant_a)
    params_b = _variant_params(variant_b)

    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(num_runs):
            shuffled = list(config["attributes"])
            random.shuffle(shuffled)
            cache_ttl_a = variant_a.get("cache_ttl", config.get("cache_ttl"))
            cache_ttl_b = variant_b.get("cache_ttl", config.get("cache_ttl"))
            messages_a = build_messages(config, shuffled, variant_a, cache_ttl_a)
            messages_b = build_messages(config, shuffled, variant_b, cache_ttl_b)
            tasks.append(api.complete(client, messages_a, **params_a, temperature=0.0))
            tasks.append(api.complete(client, messages_b, **params_b, temperature=0.0))

        if warm_cache:
            return await _gather_with_warm_cache(tasks)
        return await tqdm_asyncio.gather(*tasks)


def build_messages(
    config: dict, attributes: list[str], variant: dict, cache_ttl: int | None = None
) -> list[dict]:
    attributes_list = "\n".join(f"- {attr}" for attr in attributes)
    character_description = _render_template(
        variant.get("character_description", config.get("character_description", ""))
    )
    cache_control = {"type": "ephemeral"}
    if cache_ttl is not None:
        cache_control["ttl"] = cache_ttl
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": character_description,
                    "cache_control": cache_control,
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
        paired_diffs = [b - a for a, b in zip(vals_a, vals_b)]
        mean_diff = statistics.mean(paired_diffs) if paired_diffs else 0
        sem_diff = _calc_sem(paired_diffs)
        diffs.append((attr, mean_a, sem_a, mean_b, sem_b, mean_diff, sem_diff))
    diffs.sort(key=lambda x: abs(x[5]), reverse=True)
    return diffs


def print_results(
    diffs: list[tuple],
    variant_a: dict,
    variant_b: dict,
    prompt_cost: float,
    completion_cost: float,
):
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
        f"\nCost: ${total:.4f} (${prompt_cost:.4f} prompt + ${completion_cost:.4f} completion)"
    )


def _variant_params(variant: dict) -> dict:
    exclude = {"label", "character_description", "cache_ttl"}
    return {k: v for k, v in variant.items() if k not in exclude}


async def _gather_with_warm_cache(tasks):
    with tqdm_asyncio(total=len(tasks)) as pbar:

        async def track(coro):
            result = await coro
            pbar.update(1)
            return result

        tracked = [track(t) for t in tasks]
        first = await asyncio.gather(*tracked[:2])
        rest = await asyncio.gather(*tracked[2:])
    return list(first) + list(rest)


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
