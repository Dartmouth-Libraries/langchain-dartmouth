from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra
from langchain.retrievers.document_compressors import CrossEncoderReranker

import requests

from dataclasses import dataclass, field
import operator
from typing import List, Optional, Sequence

from dartmouth_langchain.base import AuthenticatedMixin

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
        return r.json()


class TeiCrossEncoderReranker(CrossEncoderReranker):
    """Document compressor that uses an instance of TextEmbeddingInference for reranking."""

    top_n: int = 3
    """Number of documents to return."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using TextEmbeddingInference.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        scores = self.client.rerank(query, [doc.page_content for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc for doc, _ in result[: self.top_n]]


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
