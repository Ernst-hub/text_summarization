# extend path to import from parent directory
import sys

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append("..")
print(sys.path)
# import llama_summarizer, from ollama_langchain/summarizer.py
from summarizer import llama_summarizer

app = FastAPI()


class SummaryRequest(BaseModel):
    url: str
    question: str
    retriever: str
    device: str
    model: str
    embedding_model: str


@app.post("/summarize")
async def summarize_text(request: SummaryRequest):
    """Summarize text from a URL.

    Args:
        url (str): URL to scrape.
        question (str): Question to ask the model.
        retriever (str): Retriever to use.
        device (str): Device to use.
        model (str): Model to use.
        embedding_model (str): Embedding model to use.

    Returns:
        str: Summarized text.
    """
    url = request.url
    question = request.question
    retriever = request.retriever
    device = request.device
    model = request.model
    embedding_model = request.embedding_model

    sum_text = llama_summarizer(
        url=url,
        question=question,
        retriever=retriever,
        device=device,
        model=model,
        embedding_model=embedding_model,
    )

    sum_text = (
        sum_text.scrape_text()
        .split_text()
        .instantiate_embeddings()
        .instantiate_llm()
        .instantiate_retriever()
        .instantiate_qa_chain()
        .generate()
    )
    temp = sum_text.answ["result"].strip()

    del sum_text

    return {"Summary": temp}
