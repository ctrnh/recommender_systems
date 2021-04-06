import pandas as pd
import surprise
import numpy as np

class Recommender:
    """
    A recommender taylored for a new user
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.already_recommended = set()
        self.user_id = self.dataset.add_new_user()
        self.t = 0

    def warmup_recommendations(self, K):
        """
        The user hasn't seen any joke yet. Select K warmup joke.
        (Random)
        Out:
            - jokes (list(tuple(int,str))):
                list of (id_joke, joke) pairs
        """
        # choose policy for warmup
        id_jokes = np.random.choice(self.dataset.ratings.columns, K, replace=False)
        return id_jokes
    
    def say(self, msg, ask=False):
        if ask:
            ans = input(f"{self}: {msg}\n")
            print(f"You: {ans}")
            return ans
        print(f"{self}: {msg}")

    def propose_joke(self, id_joke):
        """
        Proposes a joke, asks and registers user feedback
        """
        joke = self.dataset.get_jokes(id_jokes=id_joke) 
        self.say(joke)
        user_rating = self.say(msg="How would you rate this joke ?", ask=True)
        self.update_user_feedback(id_joke=id_joke, rating=user_rating)

    def update_user_feedback(self, id_joke, rating):
        self.already_recommended.add(id_joke)
        self.dataset.add_new_rating(user_id=self.user_id, item_id=id_joke, rating=rating)
        

    def train_predict(self):
        # eg: retrain svd with new ratings
        pass

    def recommend(self,):
        """
        recommendation of 1 joke 
        Out:
            id_joke
        """
        raise NotImplementedError
        
