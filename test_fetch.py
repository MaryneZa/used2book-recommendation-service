
from util import DataFetcher


# Test script (test_fetch.py)
if __name__ == "__main__":
    fetcher = DataFetcher()
    books = fetcher.get_all_books()
    users = fetcher.get_all_users()
    book_genres = fetcher.get_book_genres()
    user_prefs = fetcher.get_user_preferred_genres()
    user_item_matrix = fetcher.get_user_item_matrix()