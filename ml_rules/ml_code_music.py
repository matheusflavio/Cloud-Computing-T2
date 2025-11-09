import pickle
import pandas as pd
import ssl
import os  # Importar a biblioteca OS para manipulação de caminhos

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import pickle

ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv("https://homepages.dcc.ufmg.br/~cunha/hosted/cloudcomp-2023s2-datasets/2023_spotify_ds1.csv")

data = df.groupby(['pid', 'track_name'])['track_name'].count().unstack().fillna(0)

def encode(x):
	if x<=0:
		return False
	if x>=1:
		return True

data_sets = data.map(encode)

frequent_itemsets = apriori(data_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.to_pickle("./ml_rules/model_rules.pkl")