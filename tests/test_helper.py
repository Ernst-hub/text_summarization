import sys
from tests import _TEST_ROOT

sys.path.append(_TEST_ROOT)
from unittest.mock import patch
import pytest
from src.summarizer import llama_summarizer

question = "Summarize the following: "
url = "https://geohot.github.io/blog/jekyll/update/2023/08/08/a-really-big-computer.html"


@pytest.fixture(scope="session")
def summarizer():
    summarizer = llama_summarizer(url=url, question=question)
    return summarizer


@pytest.mark.usefixtures("summarizer")
class TestLogMethodCall:
    @patch("logging.info")
    def test_logging_when_verbose_is_true(self, mock_logging, summarizer):
        summarizer.verbose = True
        summarizer.generate()

        assert (
            mock_logging.call_count > 0
        ), "logging.info should be called when verbose is True"

    @patch("logging.info")
    def test_no_logging_when_verbose_is_false(
        self, mock_logging, summarizer
    ):
        summarizer.verbose = False
        summarizer.generate()

        # Check if logging.info was never called
        assert (
            mock_logging.call_count == 0
        ), "logging.info should not be called when verbose is False"
