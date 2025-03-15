from flask import Flask, request, jsonify
from util import DataFetcher
import logging
import pickle
import os

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load service from shared volume
MODEL_FILE = '/data/recommendation_service.pkl'

def load_recommendation_service():
    """Load the recommendation service from file."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['service'], data['user_item_matrix']
    except Exception as e:
        logger.error(f"Failed to load recommendation service: {e}")
        raise

recommendation_service, user_item_matrix = load_recommendation_service()
logger.info("Recommendation service loaded from file")

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        logger.warning("No user_id provided in request")
        return jsonify({"error": "user_id is required"}), 400

    user_data = DataFetcher.get_user_data(user_id)
    if not user_data:
        logger.info(f"User ID {user_id} not found")
        return jsonify({"error": "User not found"}), 404

    try:
        recommended_books = recommendation_service.get_recommendations(user_id, user_item_matrix)
        recommendations = recommended_books.to_dict(orient='records')
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)