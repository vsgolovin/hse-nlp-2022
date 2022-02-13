from os import path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import ieee


def main():
    # launch webdriver
    opts = Options()
    # opts.add_argument('--headless')
    driver = webdriver.Chrome(service=Service('/usr/bin/chromedriver'),
                              options=opts)
    driver.set_window_size(1280, 900)
    driver.implicitly_wait(10)

    # parse `All Issues` page for every journal
    with open('urls/journals.csv', 'r') as jf:
        next(jf)  # skip header
        for line in jf:
            _, alias, url = line.strip().split(',')
            outfile = f'urls/issues/{alias}.csv'
            if not path.exists(outfile):
                ieee.parse_issue_list(driver, url, outfile=outfile)

    # TODO: parse issue and article pages with selenium
    driver.quit()


if __name__ == '__main__':
    main()
