from uuid import UUID
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_dartmouth.definitions import LLM_BASE_URL
from langchain_dartmouth.base import AuthenticatedMixin

from typing import AsyncIterator, Callable, Coroutine, Dict, Iterator, List


class DartmouthLLM(HuggingFaceTextGenInference, AuthenticatedMixin):
    """
    Dartmouth-deployed Large Language Models. Use this class for non-chat models
    (e.g., [CodeLlama 13B](https://huggingface.co/meta-llama/CodeLlama-13b-hf)).

    This class does not format the prompt to adhere to any required templates.
    The string you pass to it is exactly the string received by the LLM. If the
    desired model requires a chat template (e.g.,
    [CodeLlama 13B Instruct](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf)),
    you may want to use `ChatDartmouth` instead.
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
        model_name="codellama-13b-python-hf",
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


def DartmouthChatModel(*args, **kwargs):
    from warnings import warn

    warn(
        "DartmouthChatModel is deprecated and will be removed in a future update. Use `DartmouthLLM` (as a drop-in replacement) or `ChatDartmouth` instead!"
    )
    return DartmouthLLM(*args, **kwargs)


class ChatDartmouth(ChatOpenAI, AuthenticatedMixin):
    """Dartmouth-deployed Chat models (also known as Instruct models).

    Use this class if you want to use a model that uses a chat template
    (e.g.,
    [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/meta-llama-3.1-8b-instruct)).

    All prompts are automatically formatted to adhere to the chosen model's
    chat template. If you need more control over the exact string sent to the model,
    you may want to use `DartmouthLLM` instead.
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
        model_name="llama-3-1-8b-instruct",
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
            model_name (str, optional): Name of the model to use. Defaults to "llama-3-1-8b-instruct".
            authenticator (Callable, optional): A Callable that returns a valid JWT to use for authentication.
                If specified, `dartmouth_api_key` is ignored.
            inference_server_url (str, optional): URL pointing to an inference endpoint. Defaults to "https://ai-api.dartmouth.edu/tgi/".
        """
        if inference_server_url:
            kwargs["openai_api_base"] = inference_server_url
        else:
            kwargs["openai_api_base"] = f"{LLM_BASE_URL}{model_name}/v1/"
        # For compliance, a non-null API key must be set
        kwargs["openai_api_key"] = "unused"
        super().__init__(*args, **kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.jwt_url = jwt_url
        self.authenticate(jwt_url=self.jwt_url)

    def invoke(self, *args, **kwargs) -> BaseMessage:
        """Invokes the model to get a response to a query."""
        try:
            return super().invoke(*args, **kwargs)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs) -> BaseMessage:
        """Invokes the model to get a response to a query."""
        try:
            response = await super().ainvoke(*args, **kwargs)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().ainvoke(*args, **kwargs)
            return response

    def stream(self, *args, **kwargs) -> Iterator[BaseMessageChunk]:
        try:
            for chunk in super().stream(*args, **kwargs):
                yield chunk
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            for chunk in super().stream(*args, **kwargs):
                yield chunk

    async def astream(self, *args, **kwargs) -> AsyncIterator[BaseMessageChunk]:
        try:
            async for chunk in super().astream(*args, **kwargs):
                yield chunk
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            async for chunk in super().astream(*args, **kwargs):
                yield chunk

    def generate(self, *args, **kwargs) -> LLMResult:
        try:
            return super().generate(*args, **kwargs)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs) -> LLMResult:
        try:
            response = await super().agenerate(*args, **kwargs)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().agenerate(*args, **kwargs)
            return response
