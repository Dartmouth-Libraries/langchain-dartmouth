from langchain_community.llms import HuggingFaceTextGenInference

from dartmouth_langchain.definitions import LLM_BASE_URL
from dartmouth_langchain.base import AuthenticatedMixin

from typing import Callable


class DartmouthLLM(HuggingFaceTextGenInference, AuthenticatedMixin):
    """
    Dartmouth-deployed Large Language Models. Use this class for non-chat models
    (e.g., [CodeLlama 13B](https://huggingface.co/meta-llama/CodeLlama-13b-hf)).

    This class does not format the prompt to adhere to any required templates.
    The string you pass to it is exactly the string received by the LLM. If the
    desired model requires a chat template (e.g.,
    [CodeLlama 13B Instruct](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf)),
    you may want to use `DartmouthChatModel` instead.
    """

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
        model_name="codellama-13b-hf",
        authenticator: Callable = None,
        jwt_url: str = None,
        inference_server_url: str = None,
        **kwargs,
    ):
        """
        Initializes the object

        Args:
            dartmouth_api_key (str, optional): A valid Dartmouth API key (see https://developer.dartmouth.edu/keys).
                If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
            model_name (str, optional): Name of the model to use. Defaults to "codellama-13b-hf".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
            inference_server_url (str, optional): URL pointing to an inference endpoint. Defaults to "https://ai-api.dartmouth.edu/tgi/".
        """
        if inference_server_url:
            kwargs["inference_server_url"] = inference_server_url
        else:
            kwargs["inference_server_url"] = f"{LLM_BASE_URL}{model_name}/"
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.jwt_url = jwt_url
        self.authenticate(jwt_url=self.jwt_url)

    def invoke(self, *args, **kwargs) -> str:
        """Invokes the model to get a response to a query."""
        try:
            return super().invoke(*args, **kwargs)
        except KeyError:
            self.authenticate(jwt_url=self.jwt_url)
            return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs) -> str:
        """Invokes the model to get a response to a query."""
        try:
            return super().ainvoke(*args, **kwargs)
        except KeyError:
            self.authenticate(jwt_url=self.jwt_url)
            return super().ainvoke(*args, **kwargs)
