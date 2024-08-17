Embeddings
===============================================

.. automodule:: langchain_dartmouth.embeddings
    :members:

Example
-----------------------------------------------

With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

.. code-block:: python

    from langchain_dartmouth.embeddings import DartmouthEmbeddingsModel


    embeddings = DartmouthEmbeddingsModel()

    response = embeddings.embed_query("Hello? Is there anybody in there?")

    print(response)


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


Reranking
===============================================

.. automodule:: langchain_dartmouth.retrievers.document_compressors
    :members:
    :exclude-members: TeiCrossEncoderReranker

Example
-----------------------------------------------
With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted Reranker only takes a few lines of code:

.. code-block:: python

    from langchain.docstore.document import Document

    from langchain_dartmouth.retrievers.document_compressors import DartmouthReranker


    docs = [
        Document(page_content="Deep Learning is not..."),
        Document(page_content="Deep learning is..."),
    ]
    query = "What is Deep Learning?"
    reranker = DartmouthReranker()
    ranked_docs = reranker.compress_documents(query=query, documents=docs)
    print(ranked_docs)
