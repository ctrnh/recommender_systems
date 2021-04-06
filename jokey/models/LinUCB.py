import pandas as pd
import surprise
import numpy as np
from recommender import *

class LinUCB(Recommender):
    def __init__(self, 
                dataset,
                reg_factor=1, 
                delta=.01,):
        super().__init__(self, dataset=dataset)
        
        self.reg_factor = reg_factor
        self.delta = delta
        
        self.load_arms_features()
        self.reset()

    def load_arms_features(self,svd_model_path="./svd"):
        # Load SVD model
        svd = surprise.dump.load(svd_model_path)[1]
        
        # Arms features
        self.arms_features = 
        self.n_features = 
        self.n_actions = 
        self.bound_features = np.max(np.sqrt(np.sum(np.abs(arm_features) ** 2, axis=1)))

        # Noise std
        self.noise_std = 2#noise_std

        # Bound theta
        self.bound_theta = bound_theta
    
    def reset(self):
        self.A_t_inv = 1/self.reg_factor*np.eye(self.n_features)
        self.b_t = np.zeros((self.n_features,))
        self.theta_hat = np.dot(self.A_t_inv, self.b_t)
        self.mu_hat = np.dot(self.arm_features, self.theta_hat)
        assert self.mu_hat.shape[0] == self.n_actions
        
    def alpha(self, ):
        B = self.noise_std
        d = self.n_features
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
        return action
    
    def update_user_feedback(self, id_joke, rating):
        super().update_user_feedback(id_joke=id_joke, rating=rating)

        a_t = id_joke
        r_t = self.compute_reward(id_joke=id_joke, rating=rating)
        u = self.arm_features[a_t].reshape((self.n_features,1))
        self.A_t_inv -= np.dot(self.A_t_inv,u).dot(np.dot(u.T,self.A_t_inv))/(1+u.T.dot(self.A_t_inv).dot(u))
        
        self.b_t += r_t*self.arm_features[a_t]
        self.theta_hat = np.dot(self.A_t_inv,self.b_t)
        self.mu_hat = np.dot(self.arm_features, self.theta_hat)
    
    def compute_reward(self,id_joke, rating):
        reward = rating - self.svd_bi - self.svd_mu
        return reward