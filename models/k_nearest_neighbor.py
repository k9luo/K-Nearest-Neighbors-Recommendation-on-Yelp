from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import numpy as np

class K_Nearest_Neighbor(object):
    def __init__(self):
        pass

    def train(self, matrix_train):
        self.similarity = cosine_similarity(X=matrix_train, Y=None, dense_output=True)

    def predict(self, matrix_train, k, lambda_diversity=0, lambda_serendipity=0):
        prediction_scores = []

        if lambda_serendipity != 0:
            item_pop_matrix = matrix_train.toarray().sum(axis=0)
            num_users = matrix_train.shape[0]

        if lambda_diversity != 0:
            self.item_similarity = cosine_similarity(X=matrix_train.T, Y=None, dense_output=True)
            num_items = matrix_train.shape[1]

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

            if lambda_diversity != 0:
                prediction_score = self.add_diversity(num_items, prediction_score, lambda_diversity)

            prediction_scores.append(prediction_score)

        return np.array(prediction_scores)

    def add_serendipity(self, num_users, item_pop_matrix, prediction_score, lambda_serendipity):
        serendipity = (1 - lambda_serendipity) + lambda_serendipity * np.log10(num_users/(item_pop_matrix+1))
        return prediction_score * serendipity

    def add_diversity(self, num_items, prediction_score, lambda_diversity):
        initial_item_rank_list = prediction_score.argsort()[::-1]
        mmr_scores = []
        S = []

        for n in range(num_items):
            if len(S) == 0:
                item = initial_item_rank_list[0]
                mmr_score = self.calculate_mmr_score(lambda_diversity=lambda_diversity,
                                                     sim1=prediction_score[item],
                                                     sim2=[[0]])
                mmr_scores.append(mmr_score[0])
                S.append(item)
            else:
                remaining_items = np.setdiff1d(initial_item_rank_list, S)

                sim2 = self.item_similarity[remaining_items][:,S]

                mmr_score = self.calculate_mmr_score(lambda_diversity=lambda_diversity,
                                                     sim1=prediction_score[remaining_items],
                                                     sim2=sim2)

                S.append(remaining_items[mmr_score.argmax()])
                mmr_scores.append(np.max(mmr_score))

        order = np.array(S).argsort()
        return np.array(mmr_scores)[order]

    def calculate_mmr_score(self, lambda_diversity, sim1, sim2):
        return (1-lambda_diversity) * sim1 - lambda_diversity * np.max(sim2, axis=1)

