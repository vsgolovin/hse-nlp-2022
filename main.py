from os import path, listdir, makedirs
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import ieee

WEBDRIVER_PATH = '/usr/bin/chromedriver'


def main():
    # launch webdriver
    opts = Options()
    opts.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(WEBDRIVER_PATH),
                              options=opts)
    driver.set_window_size(1280, 900)
    driver.implicitly_wait(10)

    # parse `All Issues` page for every journal
    journals_file = path.join('urls', 'journals.csv')
    with open(journals_file, 'r') as jf:
        next(jf)  # skip header
        for line in jf:
            _, alias, url = line.strip().split(',')
            outfile = path.join('urls', 'issues', alias + '.csv')
            if not path.exists(outfile):
                ieee.parse_issue_list(driver, url, outfile=outfile)

    # parse issue and article pages with selenium
    issues_dir = path.join('urls', 'issues')
    fnames = listdir(issues_dir)  # files with issues
    for fname in fnames:
        journal_name = fname.removesuffix('.csv')
        with open(path.join(issues_dir, fname), 'r') as f:
            next(f)  # skip header

            # parse every issue page
            for line in f:
                year, volume, issue, url = line.rstrip().split(',')
                url = url + '&sortType=vol-only-seq&rowsPerPage=100&pageNumber=1'
                outfile_name = f'{journal_name}_{year}_{volume}({issue}).txt'
                outfile_name = outfile_name.replace('/', '-')
                outfile_dir = path.join('database', journal_name)
                if not path.exists(outfile_dir):
                    makedirs(outfile_dir)
                outfile = path.join(outfile_dir, outfile_name)
                print(outfile)
                if path.exists(outfile) and path.getsize(outfile) > 0:
                    continue  # already parsed issue

                # parse every paper webpage
                outfile = open(outfile, 'w')
                paper_urls = []
                for _ in range(5):
                    try:
                        paper_urls = ieee.parse_issue_page(driver, url)
                        break
                    except TimeoutException:
                        print(f'Failed to load {url}')
                        sleep(10)
                if not paper_urls:
                    outfile.close()
                    raise Exception(f'Failed to load {url}')
                for paper_url in paper_urls:
                    paper = ieee.IEEEPaper(paper_url)
                    paper.get_data(driver)
                    paper.write_entry(outfile)
                outfile.close()

    driver.quit()


if __name__ == '__main__':
    main()
