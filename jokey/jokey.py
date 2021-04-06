#%%
import numpy as np
import surprise
import argparse
import logging
from datasets.jester import *
from models import SVD
from models import LinUCB
#%%

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--algo_name", type=str, default="svd_linucb")
parser.add_argument("--K_ratings", type=int, default=10)
args = parser.parse_args()

if __name__ == "__main__":
    logger.info("test log")
    print("Hi! I'm Jokey, your fun buddy! Please let me load my jokes...")

    jester = JesterDataset(jokes_path="./datasets/Dataset3JokeSet.xlsx",
                        ratings_path="./datasets/jester_dataset3.xls")


    ans = input("Do you want to see: \n 1. the most popular jokes ? \n 2. Personnalized recommendations?  \n")#\n 3. personnalized recommandations with exploration?\n")
    while ans not in {"1","2"}:
        ans = input("Please enter valid value.")
    
    if ans == "1":
        i = 0
        pop_id_jokes = jester.sort_by_popularity()
        keep_going = "y"
        while keep_going == "y":
            id = pop_id_jokes[i]
            joke = jester.get_jokes(id)
            joke_stat = jester.get_stat_joke(id)
            print(f"{joke}\n \n This joke has an average rating of {round(joke_stat['mean'],2)} among {joke_stat['count']} users")
            i += 1
            keep_going = input("Do you want another joke?[y/n] \n")
            while keep_going not in {"y","n"}:
                keep_going = input("Please enter a valid value.[y/n] \n")



    elif ans == "2":
        cold_start = False
        if args.algo_name.lower() == "svd":
            algo = SVD.SVD(dataset=jester)

        elif args.algo_name.lower() == "item_knn":
            algo = recommender.ItemKnn(dataset=jester)
        elif args.algo_name.lower() == "user_knn":
            algo = recommender.UserKnn(dataset=jester)
        elif args.algo_name.lower() == "svd_linucb":
            cold_start = True
            algo = LinUCB.LinUCB(dataset=jester,
                                        )


        
        if not cold_start:
            id_jokes = algo.warmup_recommendations(K=args.K_ratings)
            for id_joke in id_jokes:
                algo.propose_joke(id_joke=id_joke)

        keep_going = "y"
        while keep_going == "y":   
            id_joke = algo.recommend()
            algo.propose_joke(id_joke)







