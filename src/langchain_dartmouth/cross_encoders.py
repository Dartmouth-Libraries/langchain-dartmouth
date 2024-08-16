import requests

from dataclasses import dataclass, field
from typing import List

VALID_TASKS = ("feature-extraction",)


@dataclass
class TextEmbeddingInferenceClient:
    """A client to interact with an instance of TextEmbeddingInference"""

    inference_server_url: str = "http://localhost:8080/"
    headers: dict = field(default_factory=lambda: {"Content-Type": "application/json"})
    session: requests.Session = field(default_factory=lambda: requests.Session())

    def rerank(self, query: str, texts: List[str]):
        self.session.headers.update(self.headers)
        r = self.session.post(
            url=self.inference_server_url + "rerank",
            json={"query": query, "texts": texts},
        )
        r.raise_for_status()
        return [text["score"] for text in sorted(r.json(), key=lambda x: x["index"])]
