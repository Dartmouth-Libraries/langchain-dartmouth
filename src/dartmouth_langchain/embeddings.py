from langchain_community.embeddings import HuggingFaceHubEmbeddings

from typing import Callable

from dartmouth_langchain.base import AuthenticatedMixin
from dartmouth_langchain.definitions import EMBEDDINGS_BASE_URL


class DartmouthEmbeddings(HuggingFaceHubEmbeddings, AuthenticatedMixin):
    """
    Extends the LangChain class HuggingFaceHubEmbeddings for more convenient
    interaction with Dartmouth's instance of Text Embeddings Inference
    """

    authenticator: Callable = None
    dartmouth_api_key: str = None
    jwt_url: str = None

    def __init__(
        self,
        *args,
        dartmouth_api_key: str = None,
        model=None,
        model_name="bge-large-en-v1-5",
        authenticator: Callable = None,
        jwt_url: str = None,
        **kwargs,
    ):
        f"""
        Initializes the object

        Args:
            dartmouth_api_key (str, optional): A valid Dartmouth API key (see https://developer.dartmouth.edu/keys).
                If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
            model_name (str, optional): Name of the model to use. Defaults to "bge-large-en-v1-5".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
        """
        if model:
            kwargs["model"] = model
        else:
            kwargs["model"] = f"{EMBEDDINGS_BASE_URL}{model_name}/"
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)
