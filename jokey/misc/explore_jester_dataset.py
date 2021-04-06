# %%
import sys
sys.path.append("../")

#from datasets import jester
from datasets.jester import *
# %%
jd = JesterDataset(jokes_path="../datasets/Dataset3JokeSet.xlsx",
ratings_path="../datasets/jester_dataset3.xls"
)
# %%
jd.ratings.head()
# %%
jd.jokes.head()
# %%
jd.surprise_ratings_df.head()
# %%
jd.jokes_stats

# %%
jd.user_stats
# %%
jd.jokes_stats.loc["count"].describe()
# %%
jd.jokes_stats[25]
# %%
jd.get_jokes(53)
# %%
sorted_by_pop_id = jd.ratings.median().sort_values(ascending=False).index

# %%
jd.jokes_stats[sorted_by_pop_id[:10]]
# %%
jd.get_jokes(sorted_by_pop_id[0])
# %%
# Which are the jokes that are the most controversial?
# May be better to recommend jokes that are different from each other during warmup
# Recommend the ones with high variance ?


# %%
# Can we cluster the jokes?
# %%
# We could also compute SVD and similarity measure between items with svd.qi