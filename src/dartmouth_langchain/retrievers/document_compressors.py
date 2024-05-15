from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.pydantic_v1 import Extra, Field

import operator
from typing import Optional, Sequence

from dartmouth_langchain.base import AuthenticatedMixin
from dartmouth_langchain.cross_encoders import TextEmbeddingInferenceClient


class TeiCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses an instance of TextEmbeddingInference for reranking."""

    client: TextEmbeddingInferenceClient = Field(
        default_factory=lambda: TextEmbeddingInferenceClient()
    )
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
