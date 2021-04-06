import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class JesterDataset:
    def __init__(self, 
                ratings_path, 
                jokes_path,
                min_n_ratings_user=5,
                min_n_ratings_joke=20
                ):
        self.ratings_path = ratings_path
        self.jokes_path = jokes_path
        self.rating_scale = (-10, 10)

        self.min_n_ratings_user = min_n_ratings_user
        self.min_n_ratings_joke = min_n_ratings_joke

        self.load_jokes()
        self.load_ratings()
        self.format_to_surprise()

    def add_new_user(self):
        user_id = self.ratings.index[-1] + 1
        self.ratings.append(pd.DataFrame([np.nan], index=[user_id]))
        return user_id

    def add_new_rating(self, user_id, item_id, rating):
        rating = float(rating)
        self.surprise_ratings_df.append(pd.DataFrame([[user_id, item_id, rating]], columns=["userID", "itemID","rating"]))
        self.ratings.loc[user_id, item_id] = rating



    def load_jokes(self):
        logger.info("Loading jokes...")
        self.jokes = pd.read_excel(self.jokes_path, header=None)
        # TODO: name columns

    def preprocess_ratings(self):
        # Replace 99 by nan values
        self.ratings.loc[:,1:] = self.ratings.loc[:,1:].replace(99, np.nan)
        #TODO: check/replace out of range ratings 

        # Get rid of jokes which are rated by less than min_n_ratings_joke
        # TODO: min_n_ratings_joke could be taken as 25-th percentile of number of ratings for jokes
        delete_id_jokes = self.ratings.columns[self.ratings.count() < self.min_n_ratings_joke]
        self.ratings = self.ratings.drop(columns=delete_id_jokes)

        # Get rid of users which have rated less than min_n_ratings_user
        delete_id_users = self.ratings.index[self.ratings[0] < self.min_n_ratings_user]
        self.ratings = self.ratings.drop(labels=delete_id_users)
        
        # Number of ratings per user
        self.n_ratings_per_user = self.ratings[0]
        self.ratings = self.ratings.drop(columns=0)
        
    def load_ratings(self):
        logger.info("Loading ratings...")
        self.ratings = pd.read_excel(self.ratings_path, header=None)
        self.preprocess_ratings()

        # Ratings stats
        self.jokes_stats = self.ratings.describe()
        self.user_stats = self.ratings.T.describe()
     
    def format_to_surprise(self):
        formatted_ratings = self.ratings.stack().reset_index()
        formatted_ratings = formatted_ratings.rename(columns={"level_0": "userID", "level_1": "itemID", 0:"rating"})
        #formatted_ratings = formatted_ratings[(formatted_ratings["rating"] <= self.rating_scale[1] ) & (formatted_ratings["rating"] >= self.rating_scale[0]) ].reset_index()
        self.surprise_ratings_df = formatted_ratings
        # TODO: assert n_ratings users
    

    def get_jokes(self, id_jokes):
        """
        id_jokes: (int) or (list(int))
        (todo: assert id_jokes are within range)
        """
        return self.jokes.iloc[id_jokes, 0]

    def get_stat_joke(self, id_jokes):
        return self.jokes_stats[id_jokes]

    def sort_by_popularity(self):
        """
        A joke is popular if it has received "good" ratings by "many" users.

        - Its median ratings is greater than the 75th percentile of all users
        - Rated by more than xx users. (xx = 40th percentile of [nb of ratings per joke])

        ou 
        sort by ratings
        """
        sorted_by_pop_id = self.ratings.median().sort_values(ascending=False).index
        # TODO filter the ones that do not have a lot of reviews
        return sorted_by_pop_id