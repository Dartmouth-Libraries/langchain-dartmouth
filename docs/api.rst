API
===============================================

Embeddings
===============================================

.. automodule:: langchain_dartmouth.embeddings
    :members:
    :inherited-members:
    :exclude-members: Config, construct, copy, dict, json, repo_id, task, update_forward_refs, validate_environment

Example
-----------------------------------------------

With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

.. code-block:: python

    from langchain_dartmouth.embeddings import DartmouthEmbeddingsModel


    embeddings = DartmouthEmbeddingsModel()

    embeddings.embed_query("Hello? Is there anybody in there?")


Large Language Models
===============================================

.. automodule:: langchain_dartmouth.llms
    :members:

Example
-----------------------------------------------

With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

.. code-block:: python

    from langchain_dartmouth.llms import DartmouthLLM

    llm = DartmouthLLM(model_name="codellama-13b-hf")

    response = llm.invoke("Write a Python script to swap two variables."")
    print(response)

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouth

    llm = ChatDartmouth(model_name="llama-3-8b-instruct")

    response = llm.invoke("Hi there!")

    print(response.content)

.. note::

    The required prompt format is enforced automatically when you are using ``ChatDartmouth``.