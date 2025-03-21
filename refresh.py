# File: refresh.py
import logging
from service import RecommendationServiceInitializer
import pickle
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Use environment variable or default to relative path
# MODEL_DIR = '/data' if os.environ.get('DOCKER_ENV') else os.getcwd()
# MODEL_FILE = os.path.join(MODEL_DIR, 'recommendation_service.pkl')

MODEL_DIR = '/data'  # ← Force this path to make sure you're using the shared volume
MODEL_FILE = os.path.join(MODEL_DIR, 'recommendation_service.pkl')


def refresh_recommendation_service():
    try:
        logger.info("Starting recommendation service refresh...")
        initializer = RecommendationServiceInitializer()
        recommendation_service = initializer.get_service()
        
        # Ensure directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'service': recommendation_service,
                'user_item_matrix': initializer.user_item_matrix
            }, f)
        logger.info("Wrote model to %s", MODEL_FILE)
        logger.info("File modified time: %s", time.ctime(os.path.getmtime(MODEL_FILE)))
    except Exception as e:
        logger.error(f"Error refreshing recommendation service: {e}")

if __name__ == "__main__":
    refresh_recommendation_service()
