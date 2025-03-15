import requests
import pandas as pd
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
fetch_data_url = os.getenv('FETCH_DATA_URL')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    @staticmethod
    def get_all_books():
        try:
            response = requests.get(f'{fetch_data_url}/book/all-books')
            response.raise_for_status()
            raw_data = response.json()
            logger.debug("Raw books data: %s", raw_data)
            data = pd.DataFrame(raw_data['books'])
            logger.info("Fetched books data: %s rows, columns: %s", data.shape[0], list(data.columns))
            logger.debug("Books data sample:\n%s", data.head().to_string())
            return data
        except requests.RequestException as e:
            logger.error("Failed to fetch books: %s", str(e))
            raise Exception(f"Failed to fetch books: {str(e)}")
        except (KeyError, TypeError) as e:
            logger.error("Invalid books data format: %s", str(e))
            raise Exception(f"Invalid books data format: {str(e)}")

    @staticmethod
    def get_all_users():
        try:
            response = requests.get(f'{fetch_data_url}/user/all-users')
            response.raise_for_status()
            raw_data = response.json()
            logger.debug("Raw users data: %s", raw_data)
            data = pd.DataFrame(raw_data['users'])
            if not all(col in data.columns for col in ['id', 'gender']):
                logger.error("Invalid users data: missing 'id' or 'gender'. Available columns: %s", list(data.columns))
                raise ValueError("Users data must contain 'id' and 'gender'")
            data = data[['id', 'gender']]
            logger.info("Fetched users data: %s rows, columns: %s", data.shape[0], list(data.columns))
            logger.debug("Users data sample:\n%s", data.head().to_string())
            return data
        except requests.RequestException as e:
            logger.error("Failed to fetch users: %s", str(e))
            raise Exception(f"Failed to fetch users: {str(e)}")
        except (KeyError, ValueError) as e:
            logger.error("Data validation error: %s", str(e))
            raise Exception(f"Data validation error: {str(e)}")

    @staticmethod
    def get_book_genres():
        try:
            response = requests.get(f'{fetch_data_url}/book/all-book-genres')
            response.raise_for_status()
            raw_data = response.json()
            logger.debug("Raw book genres data: %s", raw_data)
            data = pd.DataFrame(raw_data['book_genres'])
            if not all(col in data.columns for col in ['book_id', 'genre_id']):
                logger.error("Invalid book genres data: missing 'book_id' or 'genre_id'. Available columns: %s", list(data.columns))
                raise ValueError("Book genres data must contain 'book_id' and 'genre_id'")
            logger.info("Fetched book genres data: %s rows, unique books: %s, unique genres: %s", 
                        data.shape[0], data['book_id'].nunique(), data['genre_id'].nunique())
            logger.debug("Book genres data sample:\n%s", data.head().to_string())
            return data
        except requests.RequestException as e:
            logger.error("Failed to fetch book genres: %s", str(e))
            raise Exception(f"Failed to fetch book genres: {str(e)}")
        except (KeyError, ValueError) as e:
            logger.error("Data validation error: %s", str(e))
            raise Exception(f"Data validation error: {str(e)}")

    @staticmethod
    def get_user_preferred_genres():
        try:
            response = requests.get(f'{fetch_data_url}/user/user-preferences')
            response.raise_for_status()
            raw_data = response.json()
            logger.debug("Raw user preferences data: %s", raw_data)
            data = pd.DataFrame(raw_data['user_preferred_genres'])
            if not all(col in data.columns for col in ['user_id', 'genre_id']):
                logger.error("Invalid user preferences data: missing 'user_id' or 'genre_id'. Available columns: %s", list(data.columns))
                raise ValueError("User preferences data must contain 'user_id' and 'genre_id'")
            logger.info("Fetched user preferred genres data: %s rows, columns: %s", data.shape[0], list(data.columns))
            logger.debug("User preferred genres data sample:\n%s", data.head().to_string())
            return data
        except requests.RequestException as e:
            logger.error("Failed to fetch user preferred genres: %s", str(e))
            raise Exception(f"Failed to fetch user preferred genres: {str(e)}")
        except (KeyError, ValueError) as e:
            logger.error("Data validation error: %s", str(e))
            raise Exception(f"Data validation error: {str(e)}")

    @staticmethod
    def get_user_item_matrix():
        try:
            response = requests.get(f'{fetch_data_url}/user/user-review')
            response.raise_for_status()
            raw_data = response.json()
            logger.debug("Raw user reviews data: %s", raw_data['reviews'])
            reviews_df = pd.DataFrame(raw_data['reviews'])
            # Aggregate duplicate reviews by averaging ratings
            aggregated_reviews = reviews_df.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()
            user_item_matrix = aggregated_reviews.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
            logger.info("Fetched user-item matrix: %s users, %s books", user_item_matrix.shape[0], user_item_matrix.shape[1])
            logger.debug("User-item matrix sample:\n%s", user_item_matrix.head().to_string())
            return user_item_matrix
        except requests.RequestException as e:
            logger.error("Failed to fetch user-item matrix: %s", str(e))
            raise Exception(f"Failed to fetch user-item matrix: {str(e)}")
        except (KeyError, TypeError) as e:
            logger.error("Invalid user-item matrix data format: %s", str(e))
            raise Exception(f"Invalid user-item matrix data format: {str(e)}")
    # @staticmethod
    # def get_user_item_matrix():
    #     try:
    #         response = requests.get(f'{fetch_data_url}/user/user-review')
    #         response.raise_for_status()
    #         raw_data = response.json()
    #         logger.debug("Raw user reviews data: %s", raw_data['reviews'][:5])
    #         reviews_df = pd.DataFrame(raw_data['reviews'])
    #         user_item_matrix = reviews_df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    #         logger.info("Fetched user-item matrix: %s users, %s books", user_item_matrix.shape[0], user_item_matrix.shape[1])
    #         logger.debug("User-item matrix sample:\n%s", user_item_matrix.head().to_string())
    #         return user_item_matrix
    #     except requests.RequestException as e:
    #         logger.error("Failed to fetch user-item matrix: %s", str(e))
    #         raise Exception(f"Failed to fetch user-item matrix: {str(e)}")
    #     except (KeyError, TypeError) as e:
    #         logger.error("Invalid user-item matrix data format: %s", str(e))
    #         raise Exception(f"Invalid user-item matrix data format: {str(e)}")



