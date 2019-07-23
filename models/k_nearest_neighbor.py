from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import numpy as np

class K_Nearest_Neighbor(object):
    def __init__(self):
        pass

    def train(self, matrix_train):
        self.similarity = cosine_similarity(X=matrix_train, Y=None, dense_output=True)

    def predict(self, matrix_train, k, lambda_serendipity=0):
        prediction_scores = []

        if lambda_serendipity != 0:
            item_pop_matrix = matrix_train.toarray().sum(axis=0)
            num_users = matrix_train.shape[0]

        for user_index in tqdm(range(matrix_train.shape[0])):
            # Get user u's similarity to all users
            vector_u = self.similarity[user_index]

            # Get closest K neighbors excluding user u self
            similar_users = vector_u.argsort()[::-1][1:k+1]

            # Get neighbors similarity weights and ratings
            similar_users_weights = self.similarity[user_index][similar_users]
            similar_users_ratings = matrix_train[similar_users].toarray()

            prediction_scores_u = similar_users_ratings * similar_users_weights[:, np.newaxis]
            prediction_score = np.sum(prediction_scores_u, axis=0)

            if lambda_serendipity != 0:
                prediction_score = self.add_serendipity(num_users, item_pop_matrix, prediction_score, lambda_serendipity)

            prediction_scores.append(prediction_score)

        return np.array(prediction_scores)

    def add_serendipity(self, num_users, item_pop_matrix, prediction_score, lambda_serendipity):
        serendipity = (1 - lambda_serendipity) + lambda_serendipity * np.log10(num_users/(item_pop_matrix+1))
        return prediction_score * serendipity

