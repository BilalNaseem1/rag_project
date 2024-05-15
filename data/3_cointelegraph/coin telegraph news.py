import csv
from newspaper import Article

# Function to get URLs from CSV
def get_urls_from_csv(csv_file, column_index):
    urls = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            urls.append(row[column_index])
    return urls

# Read URLs from the CSV file
csv_file_path = "coin_telegraph.csv"
url_column_index = 0  # Assuming the URL is in the second column (index 1)
urls = get_urls_from_csv(csv_file_path, url_column_index)

# Open CSV file in append mode
csv_output_file = 'coin_telegraph_details.csv'
with open(csv_output_file, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Process each URL
    for url in urls:
        # Download the article using Newspaper3k
        article = Article(url)
        article.download()
        article.parse()

        # Extract metadata
        article_url = url
        article_title = article.title
        article_publish_date = article.publish_date
        article_authors = ', '.join(article.authors)
        article_text = article.text

        # Write data to CSV
        writer.writerow([article_url, article_title, article_publish_date, article_authors, article_text])

print("CSV file 'article_data.csv' has been updated successfully.")