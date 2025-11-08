import pickle
import pandas as pd
import ssl

from fpgrowth_py import fpgrowth

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

# Executa FP-Growth usando as transações construídas
freqItemSet, rules = fpgrowth(transactions, minSupRatio=0.1, minConf=0.5)

# Persiste apenas as regras geradas (formato simples para consumo pelo servidor)
model_path = "./recommendation_model.pickle"
with open(model_path, 'wb') as f:
    pickle.dump(rules, f)

print(f"Modelo salvo em {model_path}")