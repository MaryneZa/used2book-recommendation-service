import logging
from service import RecommendationServiceInitializer
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = '/data/recommendation_service.pkl'

def refresh_recommendation_service():
    try:
        logger.info("Starting recommendation service refresh...")
        initializer = RecommendationServiceInitializer()
        recommendation_service = initializer.get_service()
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'service': recommendation_service,
                'user_item_matrix': initializer.user_item_matrix
            }, f)
        logger.info("Recommendation service refreshed and saved successfully")
    except Exception as e:
        logger.error(f"Error refreshing recommendation service: {e}")

if __name__ == "__main__":
    refresh_recommendation_service()