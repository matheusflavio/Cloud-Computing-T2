import pickle
import pandas as pd

from fpgrowth_py import fpgrowth

# Exemplo mínimo: carrega CSV, executa fpgrowth e salva as regras em pickle.
file_path = "../datasets/2023_spotify_ds1.csv"

# Lê o CSV em um DataFrame (operação I/O)
df = pd.read_csv(file_path)

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
model_path = "./recommend-rules/recommendation_model.pickle"
with open(model_path, 'wb') as f:
    pickle.dump(rules, f)
model_path = "./recommendation_model.pickle"
with open(model_path, 'wb') as f:
    pickle.dump(rules, f)