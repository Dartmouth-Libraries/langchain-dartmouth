import requests

from dataclasses import dataclass, field
from typing import List

VALID_TASKS = ("feature-extraction",)


@dataclass
class TextEmbeddingInferenceClient:
    """A client to interact with an instance of TextEmbeddingInference"""

    inference_server_url: str = "http://localhost:8080/"
    session: requests.Session = field(default_factory=lambda: requests.Session())

    def rerank(self, query: str, texts: List[str]):
        r = self.session.post(
            url=self.inference_server_url + "rerank",
            json={"query": query, "texts": texts},
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return [text["score"] for text in sorted(r.json(), key=lambda x: x["index"])]


# class TextEmbeddingInferenceCrossEncoder(BaseCrossEncoder):
#     """Access to reranking using an instance of HuggingFace's Text Embedding Inference"""

#     def __init__(
#         self,
#         model_name: str = "BAAI/bge-reranker-large",
#         inference_server_url: str = "https://localhost:8080/rerank",
#     ):
#         self.client = None
#         self.model_name = model_name
#         self.inference_server_url = inference_server_url

#     def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
#         """Score pairs' similarity.

#         Args:
#             text_pairs: List of pairs of texts.

#         Returns:
#             List of scores.
#         """


# class DartmouthCrossEncoder(TextEmbeddingInferenceCrossEncoder, AuthenticatedMixin):
#     """Extends the TextEmbeddingInferenceCrossEncoder class to facilitate access
#     to a Dartmouth-hosted instance of Text Embedding Inference.
#     """
