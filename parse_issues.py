import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def parse_volume_list(driver, url, outfile=None):
    def write_line(line):
        if outfile:
            f.write(line + '\n')
        else:
            print(line)

    if outfile:
        f = open(outfile, 'w')
        f.write(','.join(['Year', 'Volume', 'Issue', 'URL']))

    # open page and find table with journal volumes
    driver.get(url)
    table = driver.find_element(By.CLASS_NAME, 'issue-details-past-tabs.year')

    # iterate over years
    for element in reversed(table.find_elements(By.TAG_NAME, 'li')):
        # get year and click on the entry to get a list of volume issues
        year = element.find_element(By.TAG_NAME, 'a').text.strip()
        element.click()

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
            write_line(line)

    if outfile:
        f.close()


if __name__ == '__main__':
    WEBDRIVER_PATH = '/usr/bin/chromedriver'
    opts = Options()
    opts.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(WEBDRIVER_PATH), options=opts)
    driver.implicitly_wait(10)
    driver.set_window_size(1024, 768)
    URL = 'https://ieeexplore.ieee.org/xpl/issues?punumber=2944&isnumber=9613808'

    parse_volume_list(driver, URL)
    driver.quit()
