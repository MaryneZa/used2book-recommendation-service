import logging
from service import RecommendationServiceInitializer  # Import both

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Run the recommendation system
if __name__ == "__main__":
    initializer = RecommendationServiceInitializer()
    service = initializer.get_service()

    # Test recommendations
    user_id_new =  1# New user with no ratings
    user_id_rated = 1  # User with ratings
    logger.info("Testing recommendations...")
    print(f"\nRecommendations for new user {user_id_new}:")
    print(service.get_recommendations(user_id_new, initializer.user_item_matrix))
    