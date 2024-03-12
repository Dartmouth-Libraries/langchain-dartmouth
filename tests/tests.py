from dartmouth_langchain.llms import DartmouthChatModel
from dartmouth_langchain.embeddings import DartmouthEmbeddings


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

    llm = DartmouthChatModel(model_name="llama-2-7b-32k-instruct")
    print(
        llm.invoke(
            "[INST]\nWrite a poem about cats\n[/INST]\n\n",
            stop_sequences=["[INST]"],
        )
    )


def test_dartmouth_embeddings():
    embeddings = DartmouthEmbeddings()
    result = embeddings.embed_query("Is there anybody out there?")
    assert result


if __name__ == "__main__":
    test_dartmouth_chat()
    test_dartmouth_embeddings()
