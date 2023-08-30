#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Literal
import click
from yaspin import yaspin


from src.summarizer import llama_summarizer


@click.command()
@click.argument("url", type=str)
@click.option(
    "--question",
    default="Summarize the following: ",
    type=str,
    help="Question to ask the model.",
)
@click.option(
    "--model",
    default="summarizev2",
    type=str,
    help="Model to use for summarization.",
)
@click.option(
    "--base-url",
    default="http://localhost:11434",
    type=str,
    help="Base URL for Ollama.",
)
@click.option(
    "--verbose",
    default=False,
    type=bool,
    help="If True, print out debug information.",
)
@click.option(
    "--chunk-size",
    default=800,
    type=int,
    help="Size of chunks to split text into.",
)
@click.option(
    "--embedding-model",
    default="large",
    type=str,
    help="Embedding model to use.",
)
@click.option(
    "--retriever",
    default="default",
    type=str,
    help="Retriever to use.",
)
@click.option(
    "--device",
    default="cpu",
    type=str,
    help="Device to use.",
)
def summarize_text(
    url: str,
    question: str,
    model: str,
    base_url: str,
    verbose: bool,
    chunk_size: int,
    embedding_model: Literal["large", "small"],
    retriever: Literal["default", "SVM", "MultiQuery"],
    device: Literal["cpu", "cuda", "mps"],
) -> str:
    """Summarize text from a URL, by using the Ollama language model.

    Args:
        url (str): URL to scrape.
        question (str): Question to ask the model.
        model (str): Model to use for summarization.
        base_url (str): Base URL for Ollama.
        verbose (bool): If True, print out debug information.
        chunk_size (int): Size of chunks to split text into. *Note: set = to context window size in Ollama.*
        embedding_model (str): Embedding model to use.
    """

    s = llama_summarizer(
        url=url,
        question=question,
        model=model,
        verbose=verbose,
        base_url=base_url,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        device=device,
        retriever=retriever,
    )

    with yaspin(text="Generating summary", color="cyan") as spinner:
        s = (
            s.scrape_text()
            .split_text()
            .instantiate_embeddings()
            .instantiate_llm()
            .instantiate_retriever()
            .instantiate_qa_chain()
            .generate()
        )
        spinner.ok("✅")

    if s.answ is not None:
        answer = s.answ["result"]
    else:
        answer = "No answer generated."

    print("Answer: ", answer)
    return answer


if __name__ == "__main__":
    summarize_text()
