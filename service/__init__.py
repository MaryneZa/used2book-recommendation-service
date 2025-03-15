import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import logging
from util import DataFetcher

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, books, users, book_genres, user_prefs):
        self.books = books
        self.users = users
        self.book_genres = book_genres
        self.user_prefs = user_prefs
        self.svd_model = None
        self.book_features = None
        self.user_genre_weights = None

    def fit(self, user_item_matrix):
        logger.info("Fitting recommendation model...")
        # Prepare content-based features
        self.books['average_rating'] = pd.to_numeric(self.books['average_rating'], errors='coerce').fillna(0)
        self.books['num_ratings'] = pd.to_numeric(self.books['num_ratings'], errors='coerce').fillna(0)
        
        # Language features
        language_onehot = pd.get_dummies(self.books['language'], prefix='lang').reindex(self.books.index, fill_value=0)
        
        # Genre features: Keep as DataFrame until concatenation
        valid_book_genres = self.book_genres[self.book_genres['book_id'].isin(self.books['id'])]
        genre_matrix = valid_book_genres.pivot_table(index='book_id', columns='genre_id', aggfunc='size', fill_value=0)
        genre_onehot = pd.DataFrame(0, index=self.books['id'], columns=[f'genre_{col}' for col in genre_matrix.columns])
        genre_onehot.loc[genre_matrix.index] = genre_matrix.values
        genre_onehot = genre_onehot.reindex(self.books.index, fill_value=0)  # Keep as DataFrame
        
        # Author features
        author_onehot = pd.get_dummies(self.books['author'], prefix='author').reindex(self.books.index, fill_value=0)
        
        # Concatenate features
        features = pd.concat([self.books[['average_rating', 'num_ratings']], language_onehot, genre_onehot, author_onehot], axis=1)
        logger.debug("Feature matrix shape: %s", features.shape)
        scaler = StandardScaler()
        self.book_features = scaler.fit_transform(features)
        if self.book_features.shape[0] != len(self.books):
            logger.error("Mismatch: book_features rows (%d) != books rows (%d)", self.book_features.shape[0], len(self.books))
            raise ValueError("Feature matrix rows must match number of books")

        # Prepare user genre preferences
        all_genre_ids = self.book_genres['genre_id'].unique()
        user_genre_matrix = self.user_prefs.pivot_table(index='user_id', columns='genre_id', aggfunc='size', fill_value=0)
        self.user_genre_weights = user_genre_matrix.reindex(columns=all_genre_ids, fill_value=0).add_prefix('genre_')
        logger.debug("User genre weights shape: %s", self.user_genre_weights.shape)

        # Fit SVD for collaborative filtering
        n_users, n_books = user_item_matrix.shape
        max_components = 50
        n_components = min(max_components, min(n_users, n_books))
        if n_components < 1:
            n_components = 1
        logger.info("Setting SVD n_components to %d (users: %d, books: %d)", n_components, n_users, n_books)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_model.fit(user_item_matrix)
        self.users = self.users.set_index('id')
        self.n_books = n_books
        logger.info("Model fitting complete.")

    def compute_content_scores(self, user_id, user_item_matrix):
        has_interactions = user_id in user_item_matrix.index and np.sum(user_item_matrix.loc[user_id].values) > 0
        
        interactions = np.zeros(len(self.books))
        if has_interactions:
            user_ratings = user_item_matrix.loc[user_id]
            for book_id in user_ratings.index:
                idx = self.books.index[self.books['id'] == book_id].tolist()
                if idx:
                    interactions[idx[0]] = user_ratings[book_id]

        if has_interactions and np.sum(interactions) > 0:
            user_profile = np.dot(interactions, self.book_features) / np.sum(interactions)
            base_content_scores = cosine_similarity([user_profile], self.book_features)[0]
        else:
            base_content_scores = np.zeros(len(self.books))

        if user_id in self.user_genre_weights.index:
            user_genre_vector = self.user_genre_weights.loc[user_id].values
            genre_features = self.book_features[:, 2:2+len(self.book_genres['genre_id'].unique())]
            genre_boost = cosine_similarity([user_genre_vector], genre_features)[0]
            content_scores = base_content_scores + 0.5 * genre_boost if has_interactions else genre_boost
        else:
            content_scores = base_content_scores

        return content_scores

    def get_recommendations(self, user_id, user_item_matrix, num_recommendations=5):
        has_ratings = user_id in user_item_matrix.index and np.sum(user_item_matrix.loc[user_id].values) > 0

        if has_ratings:
            user_collab_vector = self.svd_model.transform(user_item_matrix.loc[[user_id]])
            if user_collab_vector.shape[1] == self.svd_model.components_.shape[0]:
                collab_book_vectors = self.svd_model.components_.T
                collaborative_scores = np.dot(user_collab_vector, collab_book_vectors.T)[0]
                if collaborative_scores.shape[0] < len(self.books):
                    collaborative_scores = np.pad(collaborative_scores, (0, len(self.books) - collaborative_scores.shape[0]), mode='constant')
                elif collaborative_scores.shape[0] > len(self.books):
                    collaborative_scores = collaborative_scores[:len(self.books)]
            else:
                logger.warning("SVD components mismatch (vector: %s, components: %s); falling back to content-based for user %d", 
                               user_collab_vector.shape, self.svd_model.components_.shape, user_id)
                collaborative_scores = np.zeros(len(self.books))

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
            hybrid_scores = 0.6 * collaborative_scores + 0.4 * content_scores
        else:
            content_scores = self.compute_content_scores(user_id, user_item_matrix)
            hybrid_scores = content_scores

        if has_ratings:
            rated_books = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
            for book_id in rated_books:
                idx = self.books.index[self.books['id'] == book_id].tolist()
                if idx:
                    hybrid_scores[idx[0]] = -np.inf

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