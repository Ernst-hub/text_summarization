import sys
import pytest
from tests import _TEST_ROOT
sys.path.append(_TEST_ROOT)

from src.summarizer import llama_summarizer

question = "Summarize the following: "
url = "https://geohot.github.io/blog/jekyll/update/2023/08/08/a-really-big-computer.html"


@pytest.fixture(scope="session")
def summarizer():
    summarizer = llama_summarizer(url=url, question=question)
    return summarizer


@pytest.fixture(scope="session")
def generated_summarizer():
    summarizer = (
        llama_summarizer(url=url, question=question)
        .scrape_text()
        .split_text()
        .instantiate_embeddings()
        .instantiate_llm()
        .instantiate_retriever()
        .instantiate_qa_chain()
        .generate()
    )
    return summarizer


class TestSummarizerInit:
    @pytest.mark.parametrize(
        "attribute, expected_value",
        [
            ("url", url),
            ("question", question),
            ("model", "summarizev2"),
            ("base_url", "http://localhost:11434"),
            ("verbose", False),
            ("chunk_size", 512),
            ("embedding_model", "large"),
            ("retriever", "default"),
            ("device", "cpu"),
            ("text", None),
            ("splits", None),
            ("vectorstore", None),
            ("llm", None),
            ("embeddings", None),
            ("qa_chain", None),
            ("answ", None),
        ],
    )
    def test_init(self, summarizer, attribute, expected_value):
        assert (
            getattr(summarizer, attribute) == expected_value
        ), f"{attribute} should be {expected_value}"


class TestSummarizerScraper:
    def test_scraper_text_not_none(self, generated_summarizer):
        assert (
            generated_summarizer.text is not None
        ), "Text should not be None"

    def test_scraper_text_not_empty(self, generated_summarizer):
        assert (
            generated_summarizer.text.strip() != ""
        ), "Text should not be empty"

    def test_scraper_text_starts_with_question(self, generated_summarizer):
        assert generated_summarizer.text.startswith(
            "[question:"
        ), "Text should start with '['"

    def test_scraper_text_ends_with_question(self, generated_summarizer):
        assert generated_summarizer.text.endswith(
            "]"
        ), "Text should end with ']'"


class TestSummarizerSplitter:
    def test_splitter_splits_not_none(self, generated_summarizer):
        assert (
            generated_summarizer.splits is not None
        ), "Splits should not be None"

    def test_splitter_splits_not_empty(self, generated_summarizer):
        assert (
            len(generated_summarizer.splits) > 0
        ), "Splits should not be empty"


class TestSummarizerEmbeddings:
    def test_embeddings_not_none(self, generated_summarizer):
        assert (
            generated_summarizer.embeddings is not None
        ), "Embeddings should not be None"


class TestSummarizerLLM:
    def test_llm_not_none(self, generated_summarizer):
        assert generated_summarizer.llm is not None, "LLM should not be None"


class TestSummarizerRetriever:
    def test_retriever_not_none(self, generated_summarizer):
        assert (
            generated_summarizer.retriever is not None
        ), "Retriever should not be None"


class TestSummarizerGenerate:
    def test_generate_not_None(self, generated_summarizer):
        assert (
            generated_summarizer.answ["result"].strip() != ""
        ), "Summary should not be empty"
