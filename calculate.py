import asyncio
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

import httpx
from ruamel.yaml import YAML


class ApiClient:
    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.total_cost = 0.0

    async def complete(
        self, client: httpx.AsyncClient, messages: list, **params
    ) -> str:
        response = await client.post(
            f"{self._BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"provider": {"only": ["anthropic"]}, "messages": messages, **params},
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(data["error"])
        self.total_cost += data["usage"]["cost_details"]["upstream_inference_cost"]
        return data["choices"][0]["message"]["content"]


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


def build_messages(config: dict, attributes: list[str]) -> list[dict]:
    attributes_list = "\n".join(f"- {attr}" for attr in attributes)
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": config["character_description"],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": f"Attributes:\n{attributes_list}"},
    ]
    return _add_cache_padding(messages)


def parse_response(response: str) -> dict[str, int]:
    results = {}
    pattern = re.compile(r'^(.+?):\s*(-?\d+)$')
    for line in response.strip().split("\n"):
        if not (match := pattern.match(line.strip())):
            raise ValueError(f"Unable to parse line: '{line}'")
        results[match.group(1).strip()] = int(match.group(2))
    return results


def calc_sem(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals) / math.sqrt(len(vals))


def compute_diffs(
    attributes: list[str], results_a: dict, results_b: dict
) -> list[tuple]:
    diffs = []
    for attr in attributes:
        vals_a = results_a[attr]
        vals_b = results_b[attr]
        mean_a = statistics.mean(vals_a) if vals_a else 0
        mean_b = statistics.mean(vals_b) if vals_b else 0
        sem_a = calc_sem(vals_a)
        sem_b = calc_sem(vals_b)
        paired_diffs = [b - a for a, b in zip(vals_a, vals_b)]
        mean_diff = statistics.mean(paired_diffs) if paired_diffs else 0
        sem_diff = calc_sem(paired_diffs)
        diffs.append((attr, mean_a, sem_a, mean_b, sem_b, mean_diff, sem_diff))
    diffs.sort(key=lambda x: abs(x[5]), reverse=True)
    return diffs


def print_results(diffs: list[tuple], model_a: str, model_b: str, total_cost: float):
    print(f"{'Attribute':<25} {'Diff':<15} {model_a:<20} {model_b:<20}")
    print("-" * 80)
    for attr, mean_a, sem_a, mean_b, sem_b, diff, sem_diff in diffs:
        diff_str = f"{diff:+.1f} ± {sem_diff:.2f}"
        val_a_str = f"{mean_a:.1f} ± {sem_a:.2f}"
        val_b_str = f"{mean_b:.1f} ± {sem_b:.2f}"
        print(f"{attr:<25} {diff_str:<15} {val_a_str:<20} {val_b_str:<20}")
    print(f"\nTotal cost: ${total_cost:.4f}")


async def run_comparisons(api: ApiClient, config: dict) -> list[str]:
    model_a = config["model_a"]
    model_b = config["model_b"]
    num_runs = config.get("num_runs", 1)

    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(num_runs):
            shuffled = list(config["attributes"])
            random.shuffle(shuffled)
            messages = build_messages(config, shuffled)
            tasks.append(api.complete(client, messages, model=model_a, temperature=0.0))
            tasks.append(api.complete(client, messages, model=model_b, temperature=0.0))
        return await asyncio.gather(*tasks)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <config_file>")
        sys.exit(1)

    yaml = YAML()
    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    api = ApiClient()
    responses = asyncio.run(run_comparisons(api, config))

    num_runs = config.get("num_runs", 1)
    results_a = defaultdict(list)
    results_b = defaultdict(list)
    for i in range(num_runs):
        for attr, val in parse_response(responses[i * 2]).items():
            results_a[attr].append(val)
        for attr, val in parse_response(responses[i * 2 + 1]).items():
            results_b[attr].append(val)

    diffs = compute_diffs(config["attributes"], results_a, results_b)
    print_results(diffs, config["model_a"], config["model_b"], api.total_cost)


if __name__ == "__main__":
    main()
