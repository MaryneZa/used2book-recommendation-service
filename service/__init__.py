import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from implicit.als import AlternatingLeastSquares
import logging
from util import DataFetcher

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, books, users, book_genres, user_prefs):
        self.books = books
        self.users = users
        self.book_genres = book_genres
        self.user_prefs = user_prefs
        self.als_model = None
        self.book_features = None
        self.user_genre_weights = None
        self.tfidf_matrix = None
        self.user_item_matrix = None
        self.book_ids = None

    def fit(self, user_item_matrix):
        logger.info("Fitting recommendation model...")
        self.books['average_rating'] = pd.to_numeric(self.books['average_rating'], errors='coerce').fillna(0)
        self.books['num_ratings'] = pd.to_numeric(self.books['num_ratings'], errors='coerce').fillna(0)
        
        # Language features
        language_onehot = pd.get_dummies(self.books['language'], prefix='lang').reindex(self.books.index, fill_value=0)
        
        # Genre features
        valid_book_genres = self.book_genres[self.book_genres['book_id'].isin(self.books['id'])]
        genre_matrix = valid_book_genres.pivot_table(index='book_id', columns='genre_id', aggfunc='size', fill_value=0)
        genre_onehot = pd.DataFrame(0, index=self.books['id'], columns=[f'genre_{col}' for col in genre_matrix.columns])
        genre_onehot.loc[genre_matrix.index] = genre_matrix.values
        genre_onehot = genre_onehot.reindex(self.books.index, fill_value=0)
        
        # Author features
        author_onehot = pd.get_dummies(self.books['author'], prefix='author').reindex(self.books.index, fill_value=0)
        
        # TF-IDF features
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = tfidf.fit_transform(self.books['description'].fillna(''))
        logger.debug("TF-IDF matrix shape: %s", self.tfidf_matrix.shape)
        
        # Combine features
        numeric_features = csr_matrix(self.books[['average_rating', 'num_ratings']].values)
        language_onehot = csr_matrix(language_onehot.values)
        genre_onehot = csr_matrix(genre_onehot.values)
        author_onehot = csr_matrix(author_onehot.values)
        features = hstack([numeric_features, language_onehot, genre_onehot, author_onehot, self.tfidf_matrix])
        scaler = StandardScaler(with_mean=False)
        self.book_features = scaler.fit_transform(features)
        logger.debug("Combined feature matrix shape: %s", features.shape)
        
        # User genre preferences
        all_genre_ids = self.book_genres['genre_id'].unique()
        user_genre_matrix = self.user_prefs.pivot_table(index='user_id', columns='genre_id', aggfunc='size', fill_value=0)
        self.user_genre_weights = user_genre_matrix.reindex(columns=all_genre_ids, fill_value=0).add_prefix('genre_')
        
        # Expand user_item_matrix to match all books
        full_user_item_matrix = pd.DataFrame(0, index=user_item_matrix.index, columns=self.books['id'])
        for col in user_item_matrix.columns:
            if col in full_user_item_matrix.columns:
                full_user_item_matrix[col] = user_item_matrix[col]
        self.book_ids = full_user_item_matrix.columns
        self.user_item_matrix = csr_matrix(full_user_item_matrix.values)
        
        # Fit ALS
        self.als_model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=15)
        self.als_model.fit(self.user_item_matrix)
        self.users = self.users.set_index('id')
        logger.info("Model fitting complete.")

    def compute_content_scores(self, user_id, user_item_matrix):
        has_interactions = user_id in user_item_matrix.index and user_item_matrix.loc[user_id].sum() > 0
        
        interactions = np.zeros(len(self.books))
        if has_interactions:
            user_ratings = user_item_matrix.loc[user_id]
            for book_id in user_ratings.index:
                idx = self.books.index[self.books['id'] == book_id].tolist()
                if idx:
                    interactions[idx[0]] = user_ratings[book_id]

        if has_interactions and np.sum(interactions) > 0:
            user_profile = csr_matrix(interactions).dot(self.book_features) / np.sum(interactions)
            base_content_scores = cosine_similarity(user_profile, self.book_features)[0]
        else:
            base_content_scores = np.zeros(len(self.books))

        if user_id in self.user_genre_weights.index:
            user_genre_vector = self.user_genre_weights.loc[user_id].values
            genre_features = self.book_features[:, 2:2+len(self.book_genres['genre_id'].unique())]
            genre_boost = cosine_similarity(csr_matrix(user_genre_vector), genre_features)[0]
            content_scores = base_content_scores + 0.5 * genre_boost if has_interactions else genre_boost
        else:
            content_scores = base_content_scores

        return content_scores

    def get_recommendations(self, user_id, user_item_matrix, num_recommendations=20):
        has_ratings = user_id in user_item_matrix.index and np.sum(user_item_matrix.loc[user_id].values) > 0

        if has_ratings:
            user_idx = user_item_matrix.index.get_loc(user_id)
            # ALS collaborative scores
            item_ids, als_scores = self.als_model.recommend(user_idx, self.user_item_matrix[user_idx], N=num_recommendations * 2, filter_already_liked_items=False)
            collaborative_scores = np.zeros(len(self.books))
            for book_idx, score in zip(item_ids, als_scores):
                book_id = self.book_ids[book_idx]
                idx = self.books.index[self.books['id'] == book_id].tolist()
                if idx:
                    collaborative_scores[idx[0]] = score

            # Gender boost
            user_gender = self.users.loc[user_id, 'gender']
            gender_boost = np.zeros(len(self.books))
            for book_id in user_item_matrix.columns:
                rated_users = user_item_matrix.index[user_item_matrix[book_id] > 0]
                same_gender_count = sum(1 for rater_id in rated_users if self.users.loc[rater_id, 'gender'] == user_gender)
                total_raters = len(rated_users)
                if total_raters > 0:
                    idx = self.books.index[self.books['id'] == book_id].tolist()
                    if idx:
                        gender_boost[idx[0]] = (same_gender_count / total_raters) * 0.1
            collaborative_scores += gender_boost

            content_scores = self.compute_content_scores(user_id, user_item_matrix)
            hybrid_scores = 0.7 * collaborative_scores + 0.3 * content_scores
        else:
            content_scores = self.compute_content_scores(user_id, user_item_matrix)
            hybrid_scores = content_scores * 2

        logger.debug("Hybrid scores for user %s: %s", user_id, hybrid_scores)
        recommended_indices = np.argsort(hybrid_scores)[-num_recommendations:][::-1]
        recommended_books = self.books.iloc[recommended_indices]
        logger.info("Recommendations for user %s: %s", user_id, recommended_books[['id', 'title']].to_dict('records'))
        return recommended_books[['id', 'title', 'average_rating', 'num_ratings']]

class RecommendationServiceInitializer:
    def __init__(self):
        logger.info("Initializing Recommendation Service...")
        self.books = DataFetcher.get_all_books()
        self.users = DataFetcher.get_all_users()
        self.book_genres = DataFetcher.get_book_genres()
        self.user_prefs = DataFetcher.get_user_preferred_genres()
        self.user_item_matrix = DataFetcher.get_user_item_matrix()

        self.recommendation_service = RecommendationService(self.books, self.users, self.book_genres, self.user_prefs)
        self.recommendation_service.fit(self.user_item_matrix)

    def get_service(self):
        return self.recommendation_service