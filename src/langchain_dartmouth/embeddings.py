from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from typing import Callable, List

from langchain_dartmouth.base import AuthenticatedMixin
from langchain_dartmouth.definitions import EMBEDDINGS_BASE_URL


class DartmouthEmbeddings(HuggingFaceEndpointEmbeddings, AuthenticatedMixin):
    """
    Extends the LangChain class HuggingFaceEndpointEmbeddings for more convenient
    interaction with Dartmouth's instance of Text Embeddings Inference
    """

    authenticator: Callable = None
    """A Callable returning a JSON Web Token (JWT) for authentication"""
    dartmouth_api_key: str = None
    """A Dartmouth API key (obtainable from https://developer.dartmouth.edu)"""
    jwt_url: str = None
    """URL of the Dartmouth API endpoint returning a JSON Web Token (JWT)"""
    embeddings_server_url: str = None
    """URL of the Dartmouth embeddings provider """

    def __init__(
        self,
        dartmouth_api_key: str = None,
        model_name="bge-large-en-v1-5",
        authenticator: Callable = None,
        jwt_url: str = None,
        embeddings_server_url: str = None,
    ):
        """
        Initializes the object

        Args:
            dartmouth_api_key (str, optional): A valid Dartmouth API key (see https://developer.dartmouth.edu/keys).
                If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
            model_name (str, optional): Name of the model to use. Defaults to "bge-large-en-v1-5".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
            embeddings_server_url (str, optional): URL pointing to an embeddings endpoint. Defaults to "https://ai-api.dartmouth.edu/tei/".
        """
        if embeddings_server_url:
            endpoint = f"{embeddings_server_url}{model_name}/"
        else:
            endpoint = f"{EMBEDDINGS_BASE_URL}{model_name}/"
        super().__init__(model=endpoint)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)

    def embed_query(self, text: str) -> List[float]:
        """Call out to the embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        try:
            return super().embed_query(text)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to the embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            return super().embed_documents(texts)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async Call to the embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        try:
            response = await super().aembed_query(text)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().aembed_query(text)
            return response

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async Call to the embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            response = await super().aembed_documents(texts)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().aembed_documents(texts)
            return response
