# Dartmouth LangChain

LangChain components for Dartmouth-hosted models.

## Getting started

1. Install the package:

```
pip install dartmouth-langchain
```

2. Obtain a Dartmouth API key from [developer.dartmouth.edu](https://developer.dartmouth.edu/)
3. Store the API key as an environment variable called `DARTMOUTH_API_KEY`:
```
export DARTMOUTH_API_KEY=<your_key_here>
```

## Using the library

Using a Dartmouth-hosted chat model:

```{python}
from dartmouth_langchain import DartmouthChatModel


llm = DartmouthChatModel()

llm.invoke("<s>[INST] Hi there! [/INST]")
```
> **Note:** Many chat models require the prompts to have a particular formatting to work correctly! The default model is a chat model from the Llama 2 family and thus [requires the tags shown in the example](https://gpus.llm-utils.org/llama-2-prompt-template/) above.

Using a Dartmouth-hosted embeddings model:

```{python}
from dartmouth_langchain import DartmouthEmbeddingsModel


embeddings = DartmouthEmbeddingsModel()

embeddings.embed_query("Hello? Is there anybody in there?)
```

