# %%
import numpy as np
import surprise
import argparse
import logging
from models import recommender
from jester import *
import importlib
importlib.reload(recommender)
#importlib.reload(jester)
# %%
#%%
print("Hi! I'm Jokey, your fun buddy! Please let me load my jokes...")

jester = JesterDataset(jokes_path="../Dataset3JokeSet.xlsx",
                    ratings_path="../jester_dataset3.xls")



# %%

# %%
serena = recommender.SerenaVanDerwootsen(dataset=jester)

# %%
#id_jokes = serena.warmup_recommendations(K=10)
id_jokes = [66,100,39,105,75,139,44,110,141,95]
for id_joke in id_jokes:
    serena.propose_joke(id_joke=id_joke)

# %%

float('3')

jester.ratings.loc[54905,30]#.dropna()).mean()

# %%
#jester.ratings.astype('float')
# %%
for item_id in serena.already_recommended:
    print(f"{item_id}, my rating: {jester.ratings.loc[serena.user_id, item_id]}, mean rating: {jester.ratings.loc[:,item_id].dropna().mean()}")


# %%
serena.algo = surprise.prediction_algorithms.SVD()
serena.train_predict()
#%%
def compute_user_predictions(self):
    self.user_predictions = []
    for item_id in self.dataset.ratings.columns:
        
        if item_id not in self.already_recommended:
            
            pred = self.algo.predict(uid=self.user_id, iid=item_id, ).est
            self.user_predictions.append((item_id, pred))
    self.user_predictions = sorted(self.user_predictions, key=lambda x:x[1], reverse=True)
    self.pred_ptr = 0


# %%
compute_user_predictions(self=serena)
# %%
serena.user_predictions
# %%
keep_going = "y"
n_recommended_jokes = 1
while keep_going == "y":
    if n_recommended_jokes%15==0:
        serena.say("Let me a moment to think about another joke you may like...")
        serena.train_predict()
        serena.say("Here, you may like this one!")
    id_joke = serena.recommend()[0]
    serena.propose_joke(id_joke)
    n_recommended_jokes += 1

# %%

compute_user_predictions(self=serena)





# %%
serena.user_predictions



# %%
serena.train_predict()
# %%
jester.get_jokes(74)
# %%
for i in jester.ratings.loc[serena.user_id,:].dropna().sort_values(ascending=False).index:
    print(i, "pred: ", serena.algo.predict(uid=serena.user_id, iid=i).est, ', true:', jester.ratings.loc[serena.user_id,i])
# %%
serena.algo.predict(uid=serena.user_id, iid=95)
# %%
