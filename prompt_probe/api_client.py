import os
from contextlib import asynccontextmanager
from pathlib import Path

import backoff
import hishel
import httpx
from hishel.httpx import AsyncCacheClient
from ruamel.yaml import YAML


class ApiClient:
    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self.prompt_cost = 0.0
        self.completion_cost = 0.0
        self._trace: list[dict] = []

    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=3)
    async def complete(self, messages: list, **params) -> tuple[str, bool]:
        request_body = {
            "provider": {"order": ["anthropic"]},
            "messages": messages,
            **params,
        }
        response = await self._client.post(
            f"{self._BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json=request_body,
        )
        data = response.json()
        self._trace.append({"request": request_body, "response": data})
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

    def write_trace(self, path: Path = Path("trace.yml")) -> None:
        yaml = YAML()
        with open(path, "w") as f:
            yaml.dump(self._trace, f)
        self._trace = []

    @classmethod
    @asynccontextmanager
    async def create(cls):
        async with cls._create_cached_client() as client:
            yield cls(client)

    @staticmethod
    def _create_cached_client() -> AsyncCacheClient:
        Path(".cache").mkdir(exist_ok=True)
        storage = hishel.AsyncSqliteStorage(database_path=Path(".cache/http_cache.db"))
        policy = _BodyKeyFilterPolicy()
        return AsyncCacheClient(storage=storage, policy=policy)


class _BodyKeyFilterPolicy(hishel.FilterPolicy):
    use_body_key = True
