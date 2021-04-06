# %%
import sys
sys.path.append("../")
from datasets.jester import *
import surprise

# %%
jd = JesterDataset(jokes_path="../../Dataset3JokeSet.xlsx",
ratings_path="../../jester_dataset3.xls"
)
#%%
model_path = os.path.join("../models", "svd")
logging.info(f"Loading model {model_path}...")
svd = surprise.dump.load(model_path)[1]
# %%

jd.ratings
# %%
svd.predict(uid=54904, iid=5)

# %%
idx = jd.ratings.loc[54904].dropna().index
# %%
for i in idx:
    print(i)
    print(svd.predict(uid=54904, iid=i,)) #r_ui=jd.ratings.loc[54904,i]))
# %%
jd.ratings.loc[54904]
# %%
svd.predict(uid=54904, iid=5).est
# %%

# %%
jd.ratings.head()
# %%
svd.bu[0] + svd.bi[:5] + svd.pu[0,:].dot(svd.qi[:5,:].T) + svd.default_prediction()
# %%
jd.ratings.head()

# %%
svd.predict(uid=0, iid=3, verbose=True)
# %%
print("pu",svd.pu.shape)
print("qi", svd.qi.shape)
print(svd.bu.shape)
print(svd.bi.shape)
# %%
svd.bi
# %%
jd.jokes_stats
# %%

# %%
jd.ratings.head()
# %%
jd.jokes_stats
# %%

# %%
user_0 = {k: v for k,v in svd.trainset.ur[39171]}
# %%
user_0[130]
# %%
(svd.trainset.ur[39171][130])


# %%
len(svd.trainset.ur.keys())
# %%
jd.ratings.head()
# %%
svd.trainset.to_inner_iid(5)
# %%
svd.estimate(u=39171, i=130)
# %%
svd.trainset.knows_item(140)
# %%
u = svd.trainset.to_inner_uid(0)
i = svd.trainset.to_inner_iid(8)
svd.bu[u] + svd.bi[i] + svd.pu[u,:].dot(svd.qi[i,:].T) + svd.default_prediction()
# %%
jd.ratings
# %%
svd.compute_similarities()
# %%

# %%

# %%

# %%
