from flask import Flask, request, jsonify
from util import DataFetcher
import logging
import pickle
import os

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variable or default to relative path
MODEL_DIR = '/data' if os.environ.get('DOCKER_ENV') else os.getcwd()
MODEL_FILE = os.path.join(MODEL_DIR, 'recommendation_service.pkl')

def load_recommendation_service():
    try:
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['service'], data['user_item_matrix']
    except Exception as e:
        logger.error(f"Failed to load recommendation service: {e}")
        raise

recommendation_service, user_item_matrix = load_recommendation_service()
logger.info("Recommendation service loaded from file")


def get_recommendations(user_id):
    try:
        recommended_books = recommendation_service.get_recommendations(user_id, user_item_matrix)
        recommendations = recommended_books.to_dict(orient='records')
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print(get_recommendations(2))