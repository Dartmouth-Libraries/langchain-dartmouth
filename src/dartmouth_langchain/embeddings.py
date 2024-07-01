from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from typing import Callable

from dartmouth_langchain.base import AuthenticatedMixin
from dartmouth_langchain.definitions import EMBEDDINGS_BASE_URL


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
