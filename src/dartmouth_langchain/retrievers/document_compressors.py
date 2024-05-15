from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.pydantic_v1 import Extra, Field

import operator
from typing import Callable, Optional, Sequence

from dartmouth_langchain.base import AuthenticatedMixin
from dartmouth_langchain.cross_encoders import TextEmbeddingInferenceClient
from dartmouth_langchain.definitions import RERANK_BASE_URL


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


class DartmouthReranker(TeiCrossEncoderReranker, AuthenticatedMixin):
    """Facilitates interaction with a Dartmouth-hosted instance of a Text Embedding Inference instance for reranking"""

    authenticator: Callable = None
    """A Callable returning a JSON Web Token (JWT) for authentication"""
    dartmouth_api_key: str = None
    """A Dartmouth API key (obtainable from https://developer.dartmouth.edu)"""
    jwt_url: str = None
    """URL of the Dartmouth API endpoint returning a JSON Web Token (JWT)"""

    def __init__(
        self,
        *args,
        dartmouth_api_key: str = None,
        model_name: str = "bge-reranker-v2-m3",
        authenticator: Callable = None,
        jwt_url: str = None,
        **kwargs,
    ):
        """
        Initializes the object

        Args:
            dartmouth_api_key (str, optional): A valid Dartmouth API key (see https://developer.dartmouth.edu/keys).
                If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
            model_name (str, optional): Name of the model to use. Defaults to "bge-reranker-v2-m3".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
        """
        endpoint = f"{RERANK_BASE_URL}{model_name}/"
        if "client" not in kwargs:
            kwargs["client"] = TextEmbeddingInferenceClient(
                inference_server_url=endpoint
            )
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)
