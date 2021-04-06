# %%
import surprise
import pandas as pd
import numpy as np


def format_to_surprise(ratings):
    formatted_ratings = ratings.stack().reset_index()
    formatted_ratings = formatted_ratings.rename(columns={"level_0": "userID", "level_1": "itemID", 0:"rating"})
    formatted_ratings = formatted_ratings[(formatted_ratings["rating"] <= 10) & (formatted_ratings["rating"] >= -10) ].reset_index()
    return formatted_ratings

#jokes = pd.read_excel("./Dataset3JokeSet.xlsx", header=None)

path_jester_dataset = "../jester_dataset3.xls"
ratings = pd.read_excel(path_jester_dataset, header=None)

outdated_jokes = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116]

ratings = ratings.drop(columns=outdated_jokes)


# drop user who did not rate enough jokes
#ratings = ratings.dropna(thresh=int(0.7*45),axis=0)
# remove duplicate users 

#jester_dataset = JesterDataset(path=)
#jester_df = jester_dataset.prepare_df()
jester_df = format_to_surprise(ratings)

print(jester_df.head())
reader = surprise.Reader(rating_scale=(-10, 10))

data = surprise.Dataset.load_from_df(jester_df[["userID", "itemID", "rating"]], reader)

train_set, test_set = surprise.model_selection.train_test_split(data, test_size=.25)

algo = surprise.SVD()

# %%
#surprise.dump.dump("./svd_trained", algo=algo)
test = surprise.dump.load("./svd_trained")[1]
predictions = test.test(test_set)


print(surprise.accuracy.rmse(predictions))
# %%
print(test_set[:10])
# %%

algo.fit(train_set)

predictions = algo.test(test_set)


print(surprise.accuracy.rmse(predictions))
# %%
def random_predictions(rating_scale, test_set):
    m, M = rating_scale
    rnd_predictions = []
    for (uid, iid, r_ui) in test_set:
        rnd_predictions.append(surprise.Prediction(uid, iid, r_ui, np.random.uniform(m,M), {"was_impossible": False} ))
    return rnd_predictions

# %%
import numpy as np
rnd_pred = random_predictions(rating_scale=(-10,10), test_set=test_set)
surprise.accuracy.rmse(rnd_pred)

# %%

def naive_predictions(algo, test_set):
    default_value_pred = algo.default_prediction()
    default_predictions = []
    for (uid, iid, r_ui) in test_set:
        default_predictions.append(surprise.Prediction(uid, iid, r_ui, default_pred,{'was_impossible': False}))
    return default_predictions


# %%
print(default_predictions[:3])
# %%
surprise.accuracy.rmse(default_predictions)
# %%
