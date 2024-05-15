from dartmouth_langchain.llms import DartmouthChatModel
from dartmouth_langchain.embeddings import DartmouthEmbeddings
from dartmouth_langchain.cross_encoders import TextEmbeddingInferenceClient


from langchain.docstore.document import Document


def test_dartmouth_chat():
    llm = DartmouthChatModel()
    response = llm.invoke("<s>[INST]Please respond with the single word OK[/INST]")
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


# def test_dartmouth_reranker():
#     docs = [
#         "Deep Learning is not...",
#         "Deep learning is...",
#     ]
#     cross_encoder = DartmouthCrossEncoder()
#     scores = cross_encoder.score(docs)

#     assert scores


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
