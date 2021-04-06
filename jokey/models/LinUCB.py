import pandas as pd
import surprise
import numpy as np
from .recommender import Recommender

class LinUCB(Recommender):
    def __init__(self, 
                dataset,
                reg_factor=1, 
                delta=.01,):
        super().__init__( dataset=dataset)
        
        self.reg_factor = reg_factor
        self.delta = delta
        
        self.load_arm_features()
        self.reset()
    
    def __str__(self):
        return "JokeySVDLin"

    def load_arm_features(self,svd_model_path="./svd"):
        # Load SVD model
        #self.svd = surprise.dump.load(svd_model_path)[1]
        self.svd = surprise.prediction_algorithms.SVD()
        # Train
        reader = surprise.Reader(rating_scale=self.dataset.rating_scale)
        data = surprise.Dataset.load_from_df(self.dataset.surprise_ratings_df[["userID", "itemID", "rating"]], reader)
        surprise.model_selection.cross_validate(self.svd, data, measures=["RMSE"], cv=5, verbose=True)


        # Arms features
        self.arm_features = self.svd.qi

        # !! index of arms are inner_id not raw_id
        self.d_features = self.arm_features.shape[1]
        self.n_items = self.arm_features.shape[0] 
        self.bound_features = np.max(np.sqrt(np.sum(np.abs(self.arm_features) ** 2, axis=1)))

        # Noise std
        self.noise_std = np.mean(np.std(self.svd.pu, axis=0))

        # Bound theta
        self.bound_theta = np.max(np.sqrt(np.sum(np.square(self.svd.pu),axis=1)))
    
    def reset(self):
        self.A_t_inv = 1/self.reg_factor*np.eye(self.d_features)
        self.b_t = np.zeros((self.d_features,))
        self.theta_hat = np.dot(self.A_t_inv, self.b_t)
        self.mu_hat = np.dot(self.arm_features, self.theta_hat)
        assert self.mu_hat.shape[0] == self.n_items
        
    def alpha(self, ):
        B = self.noise_std
        d = self.d_features
        L = self.bound_features
        lamb = self.reg_factor
        delta = self.delta
        alpha_t = B*np.sqrt(d*np.log((1+self.t*L/lamb)/delta)) + np.sqrt(lamb)*self.bound_theta
        return alpha_t
    
    def recommend(self, ):
        norm_phi_Ainv = np.dot(self.arm_features, self.A_t_inv).dot(self.arm_features.T)
        UCB = self.mu_hat + self.alpha()*np.sqrt(np.diag(norm_phi_Ainv))
        
        best_arms = np.argwhere(UCB == np.max(UCB)).flatten()
        action = np.random.choice(best_arms)
        self.t += 1

        id_joke = self.svd.trainset.to_raw_iid(action)
        return id_joke
    
    def update_user_feedback(self, id_joke, rating):
        rating = float(rating)
        super().update_user_feedback(id_joke=id_joke, rating=rating)

        inner_id_joke = self.svd.trainset.to_inner_iid(id_joke)
        a_t = inner_id_joke
        r_t = self.compute_reward(inner_id_joke=inner_id_joke, rating=rating)
        u = self.arm_features[a_t].reshape((self.d_features,1))
        self.A_t_inv -= np.dot(self.A_t_inv,u).dot(np.dot(u.T,self.A_t_inv))/(1+u.T.dot(self.A_t_inv).dot(u))
        
        self.b_t += r_t*self.arm_features[a_t]
        self.theta_hat = np.dot(self.A_t_inv,self.b_t)

        # once item has been recommended, set features to 0
        self.arm_features[a_t, :] = 0
        self.mu_hat = np.dot(self.arm_features, self.theta_hat)
    
    def compute_reward(self,inner_id_joke, rating):
        
        reward = rating - self.svd.bi[inner_id_joke] - self.svd.default_prediction()
        return reward