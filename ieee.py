import re
from time import sleep
from sys import stdout
from selenium.webdriver.common.by import By


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


def parse_issue_page(driver, url, outfile=None):
    if outfile:
        f = open(outfile, 'w')
    else:
        f = stdout

    # TODO: get all article URLs from the page
    driver.get(url)

    if outfile:
        f.close()
