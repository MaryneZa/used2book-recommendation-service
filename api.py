from flask import Flask, request, jsonify
from service import RecommendationServiceInitializer
from util import DataFetcher
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the recommendation service
try:
    initializer = RecommendationServiceInitializer()
    recommendation_service = initializer.get_service()
except Exception as e:
    logger.error(f"Failed to initialize recommendation service: {e}")
    raise

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """
    Get book recommendations for a given user ID.
    Query parameter: user_id (integer)
    Returns: JSON with recommendations or error message
    """
    # Get user_id from query parameter
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        logger.warning("No user_id provided in request")
        return jsonify({"error": "user_id is required"}), 400

    # # Fetch user data to validate existence
    # user_data = DataFetcher.get_user_data(user_id)
    # if not user_data:
    #     logger.info(f"User ID {user_id} not found")
    #     return jsonify({"error": "User not found"}), 404

    try:
        # Generate recommendations
        recommended_books = recommendation_service.get_recommendations(user_id, initializer.user_item_matrix)
        
        # Convert DataFrame to list of dictionaries for JSON
        recommendations = recommended_books.to_dict(orient='records')
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)