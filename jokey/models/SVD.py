import pandas as pd
import surprise
import numpy as np
from .recommender import Recommender

class SVD(Recommender):
    def __init_(self, dataset):
        
        super().__init__( dataset=dataset)

    def __str__(self):
        return "JokeySVD"

    def train_predict(self):
        self.algo = surprise.prediction_algorithms.SVD()
        # Train
        reader = surprise.Reader(rating_scale=self.dataset.rating_scale)
        data = surprise.Dataset.load_from_df(self.dataset.surprise_ratings_df[["userID", "itemID", "rating"]], reader)
        surprise.model_selection.cross_validate(self.algo, data, measures=["RMSE"], cv=5, verbose=True)
        
        # Predict user ratings
        self.compute_user_predictions()

    def compute_user_predictions(self):
        self.user_predictions = []
        for item_id in self.dataset.ratings.columns:
            if item_id not in self.already_recommended:
                pred = self.algo.predict(uid=self.user_id, iid=item_id, ).est
                self.user_predictions.append((item_id, pred))
        self.user_predictions = sorted(self.user_predictions, key=lambda x:x[1], reverse=True)
        self.pred_ptr = 0

    def recommend(self, ):
        if self.t%10 == 0:
            self.say("Let me a moment to think about what kind of jokes you may like...")
            self.train_predict()
            self.say("Here, you may like this one!")
        if len(self.user_predictions) == 0:
            print("I'm out of jokes!")
            return
        id_joke, pred_rating = self.user_predictions[self.pred_ptr]
        #self.say(f"I bet you would predict {pred_rating}")
        self.pred_ptr += 1
        self.t += 1
        return id_joke