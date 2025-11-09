import pickle
import pandas as pd
import ssl
import os  # Importar a biblioteca OS para manipulação de caminhos

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import pickle

ssl._create_default_https_context = ssl._create_unverified_context

# Exemplo mínimo: carrega CSV, executa fpgrowth e salva as regras em pickle.
df = pd.read_csv("https://homepages.dcc.ufmg.br/~cunha/hosted/cloudcomp-2023s2-datasets/2023_spotify_ds1.csv")

# Monta transações: agrupa por playlist/session quando possível, senão usa a coluna de título
if "pid" in df.columns and "track_name" in df.columns:
    transactions = df.groupby("pid")["track_name"].apply(lambda s: s.dropna().astype(str).tolist()).tolist()
elif "track_name" in df.columns:
    transactions = df["track_name"].dropna().astype(str).apply(lambda x: [x]).tolist()
else:
    # fallback simples: usa todas as colunas de texto por linha
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    transactions = df[obj_cols].apply(lambda r: [f"{c}:{r[c]}" for c in obj_cols if pd.notnull(r[c])], axis=1).tolist()
    transactions = [t for t in transactions if t]

def encode(x):
    if x <= 0:
        return False
    if x >= 1:
        return True
    
sets = transactions.applymap(encode)

frequent_itemsets = apriori(sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.to_pickle("./ml_rules/model_rules.pkl")