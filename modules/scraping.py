import os
import pandas as pd
from datetime import datetime, timezone
from google_play_scraper import Sort, reviews


class PlayStoreScraper:
    def __init__(self, app_id, release_date, end_date, csv_dir="../data"):
        self.app_id = app_id
        self.release_date = release_date
        self.end_date = end_date
        self.csv_dir = csv_dir
        self.data = []

    def scrape_reviews(self, max_reviews=10000):
        continuation_token = None

        while len(self.data) < max_reviews:
            result, continuation_token = reviews(
                self.app_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=5000,
                continuation_token=continuation_token
            )

            if not result:
                break

            for review in result:
                review_date = datetime.fromtimestamp(review['at'].timestamp(), tz=timezone.utc)

                if review_date < self.release_date:
                    return

                if self.release_date <= review_date <= self.end_date:
                    self.data.append({
                        "Author": review['userName'],
                        "Comment": review['content'],
                        "Date": review_date.strftime('%Y-%m-%d %H:%M:%S')
                    })

            if continuation_token is None:
                break

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_to_csv(self, filename="playstore_reviews.csv"):
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)

        df = self.to_dataframe()
        file_path = os.path.join(self.csv_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Saved to {file_path}")

    def run(self, max_reviews=10000, save=True):
        print("Starting PlayStore scraping...")
        self.scrape_reviews(max_reviews=max_reviews)
        print(f"Scraping completed. Total reviews collected: {len(self.data)}")
        if save:
            self.save_to_csv()


if __name__ == "__main__":
    scraper = PlayStoreScraper(
        app_id='com.mojang.minecraftpe',
        release_date=datetime(2024, 6, 13, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 3, tzinfo=timezone.utc),
        csv_dir="../data"
    )
    scraper.run()
