from langchain_community.llms import HuggingFaceTextGenInference

from dartmouth_auth.definitions import ENV_NAMES
from dartmouth_langchain.definitions import LLM_BASE_URL
from dartmouth_langchain.base import AuthenticatedMixin

from typing import Callable


class DartmouthChatModel(HuggingFaceTextGenInference, AuthenticatedMixin):
    """
    Extends the LangChain class HuggingFaceTextGenInference for more convenient
    interaction with Dartmouth's instance of Text Generation Inference
    """

    authenticator: Callable = None
    dartmouth_api_key: str = None
    jwt_url: str = None

    def __init__(
        self,
        *args,
        dartmouth_api_key: str = None,
        model_name="llama-2-13b-chat-hf",
        authenticator: Callable = None,
        jwt_url: str = None,
        inference_server_url: str = None,
        **kwargs,
    ):
        f"""
        Initializes the object

        Args:
            dartmouth_api_key (str, optional): A valid Dartmouth API key (see https://developer.dartmouth.edu/keys).
                If not specified, it is attempted to be inferred from an environment variable {ENV_NAMES["dartmouth_api_key"]}.
            model_name (str, optional): Name of the model to use. Defaults to "llama-2-13b-chat-hf".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
            inference_server_url (str, optional): URL pointing to an inference endpoint. Defaults to '{LLM_BASE_URL}{model_name}/'.
        """
        if inference_server_url:
            kwargs["inference_server_url"] = inference_server_url
        else:
            kwargs["inference_server_url"] = f"{LLM_BASE_URL}{model_name}/"
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)

    def invoke(self, *args, **kwargs) -> str:
        """Invokes the model to get a response to a query."""
        try:
            return super().invoke(*args, **kwargs)
        except KeyError:
            self.authenticate()
            return super().invoke(*args, **kwargs)
