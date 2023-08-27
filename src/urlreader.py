import nltk
import requests
from bs4 import BeautifulSoup


class Scraper:
    """Scrape text from a URL.

    Args:
        url (str): URL to scrape.
        first_three (bool): If True, only scrape the first three sentences.

    Returns:
        text (str): Text scraped from URL.
    """

    def __init__(self, url: str, **kwargs):
        self.url = url
        self.first_three = kwargs.get("first_three", False)
        self.r = None

    def get_request(self):
        """Make a GET request to the URL."""
        self.r = requests.get(self.url)
        if self.r.status_code != 200:
            raise Exception("Request failed, check URL")

    def scrape_text(self):
        """Scrape text from URL."""
        if self.r is None:
            self.get_request()
        assert self.r.status_code == 200, "Request failed, check URL"
        soup = BeautifulSoup(self.r.text, "html.parser")
        text = " ".join([p.text for p in soup.find_all("p")])
        return text

    def first_three_sentences(self):
        """Scrape only the first three sentences from the URL.
        We use this as a baseline for summarization.
        """
        text = self.scrape_text()
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentences = tokenizer.tokenize(text)
        return " ".join(sentences[:3])


if __name__ == "__main__":
    s = Scraper(
        "https://www.nytimes.com/2020/04/07/us/politics/coronavirus-trump.html",
        first_three=True,
    )
    text = s.scrape_text()
