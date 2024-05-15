from dartmouth_langchain.llms import DartmouthChatModel
from dartmouth_langchain.embeddings import DartmouthEmbeddings
from dartmouth_langchain.cross_encoders import TextEmbeddingInferenceClient
from dartmouth_langchain.retrievers.document_compressors import (
    TeiCrossEncoderReranker,
    DartmouthReranker,
)

from langchain.docstore.document import Document


def test_dartmouth_chat():
    llm = DartmouthChatModel()
    response = llm.invoke("<s>[INST]Please respond with the single word OK[/INST]")
    assert response.strip() == "OK"

    llm = DartmouthChatModel(model_name="llama-3-8b-instruct", temperature=0.01)
    response = llm.invoke(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease respond with the single word OK<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    assert response.strip() == "OK"

    llm = DartmouthChatModel(model_name="codellama-13b-instruct-hf")
    response = llm.invoke("<s>[INST]Please respond with the single word OK[/INST]")
    assert response.strip() == "OK"

    llm = DartmouthChatModel(
        inference_server_url="https://ai-api.dartmouth.edu/tgi/codellama-13b-instruct-hf/",
    )
    print(llm.invoke("<s>[INST]Hello[/INST]"))


def test_dartmouth_embeddings():
    embeddings = DartmouthEmbeddings()
    result = embeddings.embed_query("Is there anybody out there?")
    assert result


def test_dartmouth_reranker():
    docs = [
        Document(page_content="Deep Learning is not..."),
        Document(page_content="Deep learning is..."),
    ]
    query = "What is Deep Learning?"
    reranker = DartmouthReranker()
    ranked_docs = reranker.compress_documents(query=query, documents=docs)
    assert ranked_docs

    reranker = DartmouthReranker(top_n=1)
    ranked_docs = reranker.compress_documents(query=query, documents=docs)
    assert len(ranked_docs) == 1


def test_tei_reranker():
    docs = [
        Document(page_content="Deep Learning is not..."),
        Document(page_content="Deep learning is..."),
    ]
    query = "What is Deep Learning?"
    cross_encoder = TeiCrossEncoderReranker()
    ranked_docs = cross_encoder.compress_documents(query=query, documents=docs)

    assert ranked_docs


def test_tei_client():
    query = "What is Deep Learning?"
    texts = [
        "Deep Learning is not...",
        "Deep learning is...",
    ]
    tei_client = TextEmbeddingInferenceClient()
    scores = tei_client.rerank(query=query, texts=texts)

    assert scores


if __name__ == "__main__":
    test_dartmouth_chat()
    test_dartmouth_embeddings()
    test_tei_client()
    test_tei_reranker()
    test_dartmouth_reranker()
