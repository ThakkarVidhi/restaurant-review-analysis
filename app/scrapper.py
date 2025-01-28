import time
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from tqdm import tqdm
import os
import json
from flask import Flask, render_template, jsonify, request
from app.socket import emit_progress
from config import Config

app = Flask(__name__)

def scrape_reviews(url, chrome_driver_path, task_id):
    # Initialize the WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36')
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # Extract restaurant name from URL for dynamic file naming
    restaurant_name = url.split('/')[-1]
    csv_file_name = f"{task_id}.csv"
    csv_file_path = os.path.join(Config.RAW_DATA_DIR, csv_file_name)

    # Ensure the raw data directory exists
    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)

    # Data collection
    reviews_data = []

    # Open the base URL to get the total number of pages
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    # Detect the total number of pages
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    pagination_footer = soup.find('footer', {'data-test': 'reviews-pagination'})
    total_pages = 1  # Default if pagination is not found

    if pagination_footer:
        page_links = pagination_footer.find_all('a')
        page_numbers = [int(link.text.strip()) for link in page_links if link.text.strip().isdigit()]
        if page_numbers:
            total_pages = max(page_numbers)
    
    print(f"Total pages found: {total_pages}")

    # Emit initial progress (20% for overall task, 0% for sub-progress)
    emit_progress(task_id, 20, "Scraping reviews...", sub_progress=0, sub_message="Starting scraping")

    # Scrape reviews across multiple pages
    for current_page in tqdm(range(1, total_pages + 1), desc="Scraping pages"):

        print(f"Scraping page {current_page} of {total_pages}")

        page_progress = int((current_page / total_pages) * 100)  # Sub-progress (0-100) for the current page
        main_progress = 20 + int((current_page / total_pages) * 10)  # Main progress (20-30) for scraping phase

        emit_progress(
            task_id,
            main_progress,  # Main progress for scraping (20 to 30)
            "Scraping reviews...",
            sub_progress=page_progress,  # Sub-progress (0 to 100) for current page
            sub_message=f"Scraping page {current_page} of {total_pages}"
        )
        
        # Construct the URL with the page query parameter
        page_url = f"{url}?page={current_page}"
        driver.get(page_url)
        time.sleep(3)  # Wait for the page to load

        # Get page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews_section = soup.find('section', id='reviews')
        if not reviews_section:
            print("Reviews section not found.")
            break

        reviews_list = reviews_section.find('ol', {'data-test': 'reviews-list'})
        if not reviews_list:
            print("Reviews list not found.")
            break

        review_items = reviews_list.find_all('li', {'data-test': 'reviews-list-item'})
        for review in review_items:
            try:
                user_name = review.find('p', class_='_1p30XHjz2rI-').text if review.find('p', class_='_1p30XHjz2rI-') else None
                city = review.find('p', class_='POyqzNMT21k-').text if review.find('p', class_='POyqzNMT21k-') else None

                ratings = {}
                for li in review.find_all('li', class_='-k5xpTfSXac-'):
                    category = li.text.split()[0]
                    rating = li.find('span', class_='-y00OllFiMo-').text if li.find('span', class_='-y00OllFiMo-') else None
                    ratings[category] = rating

                comment = review.find('span', {'data-test': 'wrapper-tag'}).text if review.find('span', {'data-test': 'wrapper-tag'}) else None
                review_info = {
                    'user_name': user_name,
                    'city': city,
                    'overall_rating': ratings.get('Overall'),
                    'food_rating': ratings.get('Food'),
                    'service_rating': ratings.get('Service'),
                    'ambience_rating': ratings.get('Ambience'),
                    'comment': comment
                }
                reviews_data.append(review_info)
            except AttributeError as e:
                print(f"Error parsing a review: {e}")

    # Close the browser
    driver.quit()

    # Save reviews to CSV
    df = pd.DataFrame(reviews_data)
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")
    
    # Emit completion signal (Scraping phase completed and sub-progress is 100%)
    emit_progress(task_id, 30, "Scraping completed", sub_progress=100, sub_message="All pages scraped successfully.")

    return csv_file_name
