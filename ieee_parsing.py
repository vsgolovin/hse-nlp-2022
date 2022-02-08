"""
Defines `IEEEPaper` class for parsing article web pages
at `ieeexplore.ieee.org`.
"""

from collections import namedtuple
import re
from bs4 import BeautifulSoup
from ieee_loader import load_paper_page


class IEEEPaper:
    URL_PREFIX = 'https://ieeexplore.ieee.org'
    Metrics = namedtuple('Metrics', ('PaperCit', 'PatentCit', 'Views'))

    def __init__(self, url: str):
        if not url.startswith(self.URL_PREFIX):
            assert url.startswith('/document/')
            url = self.URL_PREFIX + url
        self.url = url
        self.authors = ()
        self.title = None
        self.abstract = None
        self.journal = None
        self.volume = None
        self.issue = None
        self.year = None
        self.doi = None
        self.is_open_access = False
        self.metrics = self.Metrics(0, 0, None)

    def get_data(self):
        """
        Load paper data by parsing its web page.
        """
        page = load_paper_page(self.url)
        page = BeautifulSoup(page, features='html.parser')
        self._find_title(page)
        self._find_authors(page)
        self._update_metadata(page)
        self._update_metrics(page)
        self._check_if_open_access(page)

    def write_entry(self, f):
        """
        Write database entry for paper to file `f`.
        """
        f.write('[paper]\n')
        f.write(f'title = {self.title}\n')
        f.write(f'authors = {";".join(self.authors)}\n')
        f.write(f'journal = {self.journal}\n')
        f.write(f'year = {self.year}\n')
        f.write(f'volume = {self.volume}\n')
        f.write(f'issue = {self.issue}\n')
        f.write(f'abstract = \"{self.abstract}\"\n')
        f.write(f'paper citations = {self.metrics.PaperCit}\n')
        f.write(f'patent citations = {self.metrics.PatentCit}\n')
        f.write(f'full text views = {self.metrics.Views}\n')
        f.write(f'open access = {self.is_open_access}\n')
        f.write(f'DOI = {self.doi}\n')
        f.write(f'URL = {self.url}\n')
        f.write('\n')

    def _find_title(self, page: BeautifulSoup):
        tag = page.find('h1', class_=re.compile(('^document-title')))
        if tag:
            self.title = tag.get_text()
            return True
        return False

    def _find_authors(self, page: BeautifulSoup):
        tag_authors = page.find('div', class_='authors-banner-row-middle')
        authors = []
        for tag in tag_authors.find_all('span', text=True):
            s = tag.get_text().strip()
            if len(s) > 1:
                authors.append(s)
        self.authors = tuple(authors)

    def _find_abstract(self, page: BeautifulSoup):
        for tag in page.find_all('div', class_='u-mb-1'):
            if tag.find('strong'):
                self.abstract = tag.div.get_text()
                return True
        return False

    def _update_metadata(self, page: BeautifulSoup):
        main_tag = page.find('section',
                             class_='document-abstract document-tab')
        if not main_tag:
            return False

        # abstract
        self._find_abstract(main_tag)

        # "Published in:" line
        published_in = main_tag.find(
            'div', class_='u-pb-1 stats-document-abstract-publishedIn')
        self.journal = published_in.find(
            'a', class_='stats-document-abstract-publishedIn'
            ).get_text().strip()
        issue_tag = published_in.find(
            'a', class_='stats-document-abstract-publishedIn-issue')
        if issue_tag:
            self.issue = re.match(
                'Issue: ([\S]+)', issue_tag.get_text()).group(1)
        # find volume number and year
        for tag in published_in.find_all('span', text=True):
            s = tag.get_text()
            m = re.search('Volume:\s*([\S]+)', s)
            if m:
                self.volume = m.group(1)
            m = re.search('(?:\s+|^)([12][0-9]{3})(?:[\s,.]+|$)', s)
            if m:
                self.year = int(m.group(1))

        # DOI
        self.doi = main_tag.find(
            'a', href=re.compile('https://doi.org/\S+$')).get_text()

    def _update_metrics(self, page: BeautifulSoup):
        metrics = [0, 0, None]

        # find <div> block with paper metrics
        met_block = page.find('div',
                              class_='document-banner-metric-container row')
        if not met_block:
            return False

        # go through buttons inside found block
        for button in met_block.find_all('button'):
            text_vals = [elem.get_text() for elem in button]
            assert len(text_vals) >= 2 and text_vals[0].isnumeric()
            num = int(text_vals[0])
            annotation = ' '.join(text_vals[1:]).lower()
            if 'paper citations' in annotation:
                metrics[0] = num
            elif 'patent citations' in annotation:
                metrics[1] = num
            elif re.match('.*full( )?text views.*', annotation):
                metrics[2] = num
        self.metrics = self.Metrics(*metrics)  # whoa
        return True

    def _check_if_open_access(self, page: BeautifulSoup):
        tag = page.find('div', class_='document-banner-access')
        if tag and tag.find('span', text='Open Access'):
            self.is_open_access = True


if __name__ == '__main__':
    paper = IEEEPaper('https://ieeexplore.ieee.org/document/1474445')
    paper.get_data()
    with open('test.csv', 'w') as f:
        paper.write_entry(f)
