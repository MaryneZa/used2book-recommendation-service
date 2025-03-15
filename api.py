from flask import Flask, request, jsonify
from service import RecommendationServiceInitializer
from util import DataFetcher
app = Flask(__name__)

# Initialize the recommendation service via the initializer
initializer = RecommendationServiceInitializer()
recommendation_service = initializer.get_service()

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    
    # Fetch the user data (if needed)
    user_data = DataFetcher.get_user_data(user_id)
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    
    # Generate recommendations
    recommended_books = recommendation_service.get_recommendations(user_id, initializer.user_item_matrix)
    
    return jsonify({"recommendations": recommended_books.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)
