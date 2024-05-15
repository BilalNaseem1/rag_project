import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Define the URL and XPath
url = "https://cointelegraph.com/category/latest-news"
# Define a more robust XPath
xpath = '//a[@class="post-card-inline__title-link"]'
print(xpath)

# Initialize the Chrome webdriver
driver = webdriver.Chrome()

# Open the URL
driver.get(url)

# Function to scroll down the page
def scroll_down():
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def scrape_hrefs():
    elements = driver.find_elements(By.XPATH, xpath)
    print("Number of elements found:", len(elements))  # Add this line
    hrefs = [element.get_attribute("href") for element in elements]
    print("Number of hrefs scraped:", len(hrefs))  # Add this line
    return hrefs

# Scroll down until the end of the page
scroll_down()

# Scrape hrefs
hrefs = scrape_hrefs()
print("Total hrefs scraped initially:", len(hrefs))  # Add this line


# Export to CSV
with open('coin_telegraph.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Href'])
    for href in hrefs:
        writer.writerow([href])

# Repeat the process
while True:
    scroll_down()
    hrefs = scrape_hrefs()
    with open('coin_telegraph.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for href in hrefs:
            writer.writerow([href])

# Close the webdriver
driver.quit()
