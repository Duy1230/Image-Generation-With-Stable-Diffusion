import os
import time
import requests  # Used by helper functions called later, included for context
from io import BytesIO  # Used by helper functions
from PIL import Image  # Used by helper functions
import pandas as pd  # Used for saving metadata later
import hashlib  # Used by helper functions
from tqdm import tqdm

# Selenium specific imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_image_links_from_page(page_url, driver):
    """
    Navigates to a page URL and extracts image source URLs and titles
    for emojis using Selenium.
    """
    driver.get(page_url)
    links = []
    try:
        # Wait for the main container of emojis to be present
        container = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "div.FS5UE28h.container")
        ))

        # Wait for at least one emoji item within the container to be present
        # Using a more specific selector for individual emoji items
        image_items = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.LQY5mtmC div.aLnnpRah.text-center")
        ))

        # Find the image div within each item
        for item in image_items:
            try:
                # Find the specific div containing the img tag
                img_container_div = item.find_element(
                    By.CSS_SELECTOR, "div.Mw1EAtrx")
                # Find the img tag within that div
                img_tag = img_container_div.find_element(By.TAG_NAME, "img")

                img_url = img_tag.get_attribute("src")
                img_title = img_tag.get_attribute("title")

                if img_url:
                    links.append((img_url, img_title))
            except Exception as e_inner:
                print(f"  Error processing one image item: {e_inner}")
                continue  # Skip to the next item if error finding img

    except Exception as e:
        print(f"Error finding/processing image items on page {page_url}: {e}")

    return links


def hash_image_content(url):
    """Calculates the MD5 hash of image content from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            return hashlib.md5(response.content).hexdigest()
        else:
            print(
                f"Error downloading image from {url}; status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error with the image download for {url}: {e}")
        return None


def convert_webp_to_jpg(webp_data):
    """Converts image data from WebP format to JPG."""
    try:
        img = Image.open(BytesIO(webp_data))
        if img.format == 'WEBP':
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()
        else:
            # If not WEBP, return original data (assuming it might be JPG/PNG already)
            return webp_data
    except Exception as e:
        print(f"Error converting WebP to JPG: {e}")
        # Return original data on error
        return webp_data


def download_image(img_url, img_name, folder_path):
    """Downloads an image, converts if WebP, and saves it."""
    try:
        response = requests.get(img_url, stream=True, timeout=15)
        if response.status_code == 200:
            # Name already includes .jpg
            img_path = os.path.join(folder_path, f"{img_name}")
            img_data = convert_webp_to_jpg(response.content)
            if img_data:  # Ensure conversion didn't fail
                with open(img_path, "wb") as f:
                    f.write(img_data)
            else:
                print(f"Failed to get valid image data for {img_url}")
        else:
            print(
                f"Error downloading image from {img_url}; status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error with the image download for {img_url}: {e}")


def process_image_page(image_url, img_title, folder_path, idx, tag, seen_hashes):
    """
    Processes a single image URL: checks for duplicates using hash,
    downloads if new, and returns metadata.
    """
    img_hash = hash_image_content(image_url)
    if img_hash and img_hash not in seen_hashes:
        seen_hashes.add(img_hash)
        # Use a consistent naming scheme, e.g., tag_index.jpg
        new_file_name = f"{tag}_{idx:07d}.jpg"
        download_image(image_url, new_file_name, folder_path)
        metadata = {
            "file_name": new_file_name,
            "image_url": image_url,
            "image_title": img_title,
            "tag": tag
        }
        return metadata
    elif img_hash is None:
        print(f"Could not hash image: {image_url}")
        return None  # Indicate failure
    else:
        # print(f"Duplicate image skipped (hash exists): {image_url}")
        return None  # Indicate duplicate/skip


def loop_over_pages(base_url, tags, total_pages, driver, folder_path):
    """
    Loops through specified tags and pages on the site, extracts image links
    using Selenium, processes them, and collects metadata.
    """
    # Ensure the target directory exists
    os.makedirs(folder_path, exist_ok=True)

    all_metadata = []
    seen_hashes = set()  # Keep track of processed image hashes to avoid duplicates

    for tag in tags:
        print(f"Processing tag: {tag}")
        tag_image_urls = []  # Collect all URLs for the tag first

        # --- Scrape URLs using Selenium ---
        for page in tqdm(range(1, total_pages + 1), desc=f"Extracting URLs for {tag}", unit="page"):
            page_url = f"{base_url}/emoji-list/tag/{tag}?page={page}"
            print(f" Scraping page: {page_url}")
            images_on_page = get_image_links_from_page(page_url, driver)
            if images_on_page:
                tag_image_urls.extend(images_on_page)
            else:
                print(
                    f" No images found on page {page} for tag {tag}, stopping tag processing.")
                # Optional: You might want to break here if a page returns no images
                # break
            # Be polite to the server, wait between page requests
            time.sleep(1)

        # --- Process Collected URLs ---
        print(
            f"\nProcessing {len(tag_image_urls)} images found for tag '{tag}'...")
        metadata_list_for_tag = []
        for idx, (img_url, img_title) in enumerate(tqdm(tag_image_urls, desc=f"Processing images for {tag}", unit="image"), start=1):
            metadata = process_image_page(
                img_url, img_title, folder_path, idx, tag, seen_hashes)
            if metadata:
                metadata_list_for_tag.append(metadata)

        all_metadata.extend(metadata_list_for_tag)
        print(
            f"Finished processing tag: {tag}. Collected {len(metadata_list_for_tag)} new image metadata entries.")

    return all_metadata, seen_hashes  # Return collected metadata and hashes


def save_metadata(metadata_list, metadata_file):
    """Saves the collected metadata list to a CSV file."""
    if metadata_list:  # Check if list is not empty
        df = pd.DataFrame(metadata_list)
        df.to_csv(metadata_file, index=False, encoding="utf-8")
        print(f"Metadata saved to {metadata_file}")
    else:
        print("No metadata collected to save.")


if __name__ == "__main__":
    WEBDRIVER_DELAY_TIME_INT = 20
    TIMEOUT_INT = 20

    service = Service()
    chrome_options = webdriver.ChromeOptions()
    # You can comment out '--headless' if you want to see the browser window
    chrome_options.add_argument("--headless")
    # Often needed in non-GUI environments
    chrome_options.add_argument("--no-sandbox")
    # Overcomes limited resource problems
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("window-size=1920x1080")

    # Initialize the driver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(TIMEOUT_INT)  # Implicit wait
    # Explicit wait object
    wait = WebDriverWait(driver, WEBDRIVER_DELAY_TIME_INT)

    # --- Configuration ---
    CRAWL_FOLDER = "crawled_data"  # Main folder for crawled data
    IMAGE_FOLDER = os.path.join(CRAWL_FOLDER, "images")  # Subfolder for images
    METADATA_FILE = os.path.join(
        CRAWL_FOLDER, "metadata.csv")  # File for metadata

    BASE_URL = "https://discords.com"
    TAGS_TO_CRAWL = ["Blobs"]  # Define the tags you want to crawl
    # Adjust how many pages per tag to crawl (be mindful of server load)
    TOTAL_PAGES_PER_TAG = 5

    # --- Execution ---
    print("Starting data collection process...")

    # Ensure the main crawl folder exists
    os.makedirs(CRAWL_FOLDER, exist_ok=True)

    # Run the main loop
    collected_metadata, final_hashes = loop_over_pages(
        base_url=BASE_URL,
        tags=TAGS_TO_CRAWL,
        total_pages=TOTAL_PAGES_PER_TAG,
        driver=driver,
        folder_path=IMAGE_FOLDER
    )

    # Save the collected metadata
    save_metadata(collected_metadata, METADATA_FILE)

    # --- Cleanup ---
    # Important: Close the browser window and quit the driver
    driver.quit()

    print("\n-------------------------------------")
    print("Data collection finished.")
    print(f"Total unique images processed: {len(final_hashes)}")
    print(f"Total metadata entries saved: {len(collected_metadata)}")
    print(f"Images saved in: {IMAGE_FOLDER}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print("-------------------------------------")
