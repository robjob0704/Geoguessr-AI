
# file that we will use to call functions in other directories that will initiate different sequences of training or
# data collection

# currently just proof of concept AI code that we are able to retrieve coordinates from
# the results page of a geoguessr game. these can be assigned to screenshotted images
# after each game played. I'm thinking when we save each screenshotted image we can just name the image
# the coordinates that the image was taken at

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re

# Set up headless mode
chrome_options = Options()
chrome_options.add_argument("--headless=new")  # Runs without opening a window
chrome_options.add_argument("--disable-gpu")  # Helps prevent some rendering issues

# Launch browser in headless mode
service = Service()  # This uses the default ChromeDriver path
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.get("https://www.geoguessr.com/game/hanWjirZel9UGTKn")

# Try extracting the correct location marker
try:
    correct_marker = driver.find_element(By.CSS_SELECTOR, '[data-qa="correct-location-marker"]')

    # Check if it contains useful data
    location_style = correct_marker.get_attribute("style")
    print("Marker Style:", location_style)
except:
    print("Correct location marker not found!")

# Alternative: Extract coordinates from Google Maps link
try:
    google_maps_link = driver.find_element(By.XPATH, '//a[contains(@href, "google.com/maps")]')
    map_url = google_maps_link.get_attribute("href")

    match = re.search(r'@([-?\d.]+),([-?\d.]+)', map_url)
    if match:
        latitude, longitude = match.groups()
        print(f"Latitude: {latitude}, Longitude: {longitude}")
    else:
        print("Could not extract coordinates from URL.")
except:
    print("Google Maps link not found.")

driver.quit()
