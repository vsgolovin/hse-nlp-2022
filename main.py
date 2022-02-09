from os import path
from bs4 import BeautifulSoup
from collections import namedtuple
from ieee_parsing import IEEEPaper
from ieee_loader import load_paper_list_page


JournalIssue = namedtuple('JournalIssue',
                          ('name', 'year', 'vol', 'issue', 'url'))
URL_POSTFIX = '&sortType=vol-only-seq&rowsPerPage=100&pageNumber=1'
INPUT_FILE = 'journal_info.csv'
OUTPUT_DIR = 'database'


def get_journal_issue():
    with open(INPUT_FILE, 'r') as f:
        next(f)
        for line in f:
            lst = line.strip().split(',')
            lst[-1] = lst[-1] + URL_POSTFIX
            yield JournalIssue(*lst)


def main():
    for JI in get_journal_issue():
        # get new journal entry
        fname = f'{JI.name}_{JI.year}_{JI.vol}({JI.issue}).txt'.replace('/', '-')
        fname = path.join(OUTPUT_DIR, fname)
        print(fname)  # budget progress bar
        if path.exists(fname):  # already parsed this page
            continue

        # collect data
        page = load_paper_list_page(JI.url)
        page = BeautifulSoup(page, features='html.parser')
        papers = []
        for paper_tag in page.find_all('div', class_='col result-item-align'):
            authors = paper_tag.find('xpl-authors-name-list').get_text()
            if not authors:  # not a "paper" -- table of contents, intro, etc.
                continue
            paper_link_tag = paper_tag.find('h2').find('a')
            if paper_link_tag:
                paper = IEEEPaper(paper_link_tag['href'])
                paper.get_data()  # open article webpage and parse it
                papers.append(paper)

        # write data to the database
        outfile = open(fname, 'w')
        for paper in papers:
            paper.write_entry(outfile)
        outfile.close()


if __name__ == '__main__':
    main()
