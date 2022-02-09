"""
Defines methods for loading webpages from `ieeexplore.ieee.org`.
"""

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

WEBDRIVER_PATH = '/usr/bin/geckodriver'


def load_paper_list_page(url: str, max_wait_time: int = 10) -> str:
    # load web driver
    opts = Options()
    opts.add_argument('--headless')
    driver = webdriver.Firefox(service=Service(WEBDRIVER_PATH),
                               options=opts)

    # load the page
    driver.get(url)
    try:
        # wait until list of papers is loaded
        WebDriverWait(driver, max_wait_time).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div.List-results-items')))
    finally:
        src = driver.page_source
        driver.quit()

    return src


def load_paper_page(url: str, max_wait_time: int = 10) -> str:
    # load web driver
    opts = Options()
    opts.add_argument('--headless')
    driver = webdriver.Firefox(service=Service(WEBDRIVER_PATH),
                               options=opts)

    # load the page
    driver.get(url)
    try:
        # wait until abstract is loaded
        WebDriverWait(driver, max_wait_time).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div.abstract-text.row')))
    except TimeoutException:
        driver.quit()
        return None
    finally:
        src = driver.page_source
        driver.quit()

    return src
