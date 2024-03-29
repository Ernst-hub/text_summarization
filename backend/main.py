import sys

from fastapi import FastAPI
from pydantic import BaseModel
import sys

sys.path.append("..")
from src.summarizer import llama_summarizer

app = FastAPI()

class SummaryRequest(BaseModel):
    url: str
    question: str
    retriever: str
    device: str
    model: str
    embedding_model: str

def run_summarizer(url, question, retriever, device, model, embedding_model):
    summarizer = llama_summarizer(
        url=url,
        question=question,
        retriever=retriever,
        device=device,
        model=model,
        embedding_model=embedding_model,
    )
    summarizer.generate()
    return summarizer.answ["result"].strip()

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

    sum_text = run_summarizer(
            url=request.url,
            question=request.question,
            retriever = request.retriever,
            device = request.device,
            model = request.model,
            embedding_model = request.embedding_model
    )

    return {"Summary": sum_text}
