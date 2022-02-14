import re
from time import sleep
from sys import stdout
from collections import namedtuple
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup


def parse_issue_list(driver, url, outfile=None, wait_time=10):
    if outfile:
        f = open(outfile, 'w')
    else:
        f = stdout
    f.write(','.join(['Year', 'Volume', 'Issue', 'URL']) + '\n')

    # open page and find table with journal volumes
    driver.get(url)
    table = driver.find_element(By.CLASS_NAME, 'issue-details-past-tabs.year')

    # accept cookies if needed
    popup = driver.find_element(By.CLASS_NAME, 'cc-compliance')
    if popup.is_displayed():
        popup.click()

    # iterate over years
    for element in reversed(table.find_elements(By.TAG_NAME, 'li')):
        # get year and click on the entry to get a list of volume issues
        anchor = element.find_element(By.TAG_NAME, 'a')
        year = anchor.text.strip()
        href = anchor.get_attribute('href')
        if href:  # only one volumme -> URL leads directly to volume page
            line = ','.join([year, '', '', href])
            f.write(line + '\n')
            continue
        element.click()
        sleep(0.1)

        # find volume number
        issue_list = driver.find_element(By.CLASS_NAME, 'u-mt-2.issue-list')
        volume = issue_list.find_element(By.TAG_NAME, 'b').text
        volume = re.search('Volume (\S+)', volume).group(1)

        # find volume issues and their URLs
        for issue_element in reversed(issue_list.find_elements(By.TAG_NAME,
                                                               'a')):
            issue = issue_element.text
            issue = re.search('Issue (\S+)', issue).group(1)
            href = issue_element.get_attribute('href')
            line = ','.join([year, volume, issue, href])
            f.write(line + '\n')

    if outfile:
        f.close()


def parse_issue_page(driver, url, wait_time=10):
    # load web page
    driver.get(url)
    try:
        # wait until list of papers is loaded
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div.List-results-items')))
    finally:
        page = BeautifulSoup(driver.page_source, features='html.parser')

    # collect paper URLs
    papers = []
    for paper_tag in page.find_all('div', class_='col result-item-align'):
        authors = paper_tag.find('xpl-authors-name-list').get_text()
        if not authors:  # not a "paper" -- table of contents, intro, etc.
            continue
        paper_anchor = paper_tag.find('h2').find('a')
        if paper_anchor and paper_anchor.has_attr('href'):
            papers.append(paper_anchor['href'])
    return papers


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

    def _load_page(self, driver, wait_time=10):
        driver.get(self.url)
        try:
            # wait until abstract is loaded
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div.abstract-text.row')))
        except TimeoutException:
            return None
        return BeautifulSoup(driver.page_source, features='html.parser')

    def get_data(self, driver, wait_time=10, attempts=5, sleep_time=10):
        """
        Load paper data by parsing its web page.
        """
        for _ in range(attempts):
            page = self._load_page(driver, wait_time)
            if page is not None:
                break
            sleep(sleep_time)
        if page is None:
            print('Could not load page {self.url}.')
            return

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
            text_vals = [elem.get_text() for elem in button.find_all('div')]
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
