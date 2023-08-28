import sys

from tests import _TEST_ROOT

sys.path.append(_TEST_ROOT)
from src.urlreader import Scraper

url = "https://geohot.github.io/blog/jekyll/update/2023/08/08/a-really-big-computer.html"


class TestScraper:
    s = Scraper(url=url)

    def test_get_request(self):
        self.s.get_request()
        assert self.s.r.status_code == 200, "Request failed, check URL"

    def test_scrape_text(self):
        assert self.s.scrape_text() is not None, "Text should not be None"
        assert self.s.scrape_text().strip() != "", "Text should not be empty"
