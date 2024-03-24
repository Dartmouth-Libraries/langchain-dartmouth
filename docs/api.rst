API
===============================================

Embeddings
===============================================

.. automodule:: dartmouth_langchain.embeddings
    :members:
    :inherited-members:
    :exclude-members: Config, construct, copy, dict, json, repo_id, task, update_forward_refs, validate_environment

Example
-----------------------------------------------

With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

.. code-block:: python

    from dartmouth_langchain import DartmouthEmbeddingsModel


    embeddings = DartmouthEmbeddingsModel()

    embeddings.embed_query("Hello? Is there anybody in there?")


Large Language Models
===============================================

.. automodule:: dartmouth_langchain.llms
    :members:

Example
-----------------------------------------------

With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

.. code-block:: python

    from dartmouth_langchain import DartmouthChatModel


    llm = DartmouthChatModel()

    llm.invoke("<s>[INST] Hi there! [/INST]")

.. note::

    Many chat models require the prompts to have a particular formatting to work correctly! The default model is a chat model from the Llama 2 family and thus `requires the tags shown in the example <https://gpus.llm-utils.org/llama-2-prompt-template/>`_ above.