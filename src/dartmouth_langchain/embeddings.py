from langchain_community.embeddings import HuggingFaceHubEmbeddings

import dartmouth_auth

from typing import Callable


class DartmouthEmbeddings(HuggingFaceHubEmbeddings):
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
        """
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
            kwargs["model"] = f"https://ai-api.dartmouth.edu/tei/{model_name}/"
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)

    def authenticate(self, jwt_url=None):
        if self.authenticator:
            jwt = self.authenticator()
        else:
            jwt = dartmouth_auth.get_jwt(
                dartmouth_api_key=self.dartmouth_api_key, jwt_url=jwt_url
            )
        self.client.headers = {"Authorization": f"Bearer {jwt}"}
