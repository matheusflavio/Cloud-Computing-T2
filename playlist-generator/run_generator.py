"""Gerador de regras de playlist.

Comentários e explicações são mantidos próximos às operações relevantes
para facilitar leitura e manutenção.
"""
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from fpgrowth_py import fpgrowth


def detect_columns(df: pd.DataFrame):
    # candidatos comuns para agrupar e para o nome da faixa
    group_candidates = ["session_id", "session", "user_id", "user", "playlist_id", "playlist"]
    item_candidates = ["track_id", "track_uri", "track_name", "song", "title", "artist_name", "name"]
    group_col = next((c for c in group_candidates if c in df.columns), None)
    item_col = next((c for c in item_candidates if c in df.columns), None)
    return group_col, item_col


def build_transactions(df: pd.DataFrame, group_col: str, item_col: str):
    # monta transações como listas de strings
    if group_col and item_col and group_col in df.columns and item_col in df.columns:
        tx = df.groupby(group_col)[item_col].apply(lambda s: s.dropna().astype(str).tolist()).tolist()
    elif item_col and item_col in df.columns:
        tx = df[item_col].dropna().astype(str).apply(lambda x: [x]).tolist()
    else:
        # fallback: coleciona colunas texto por linha
        obj_cols = df.select_dtypes(include=["object", "category"]).columns
        tx = df[obj_cols].apply(lambda r: [f"{c}:{r[c]}" for c in obj_cols if pd.notnull(r[c])], axis=1).tolist()
        tx = [t for t in tx if t]
    return tx


def main():
    parser = argparse.ArgumentParser(description="Run FP-Growth on a CSV dataset and save rules to a pickle file")
    parser.add_argument("--csv", required=True, help="Path to CSV file with song records (mounted into the container)")
    parser.add_argument("--out", default="/output/recommendation_model.pickle", help="Output pickle path")
    parser.add_argument("--minsup", type=float, default=0.02, help="Minimum support ratio (e.g. 0.02)")
    parser.add_argument("--minconf", type=float, default=0.5, help="Minimum confidence (e.g. 0.5)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    # Validação básica do caminho do CSV antes de prosseguir
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Carrega o CSV em um DataFrame (pandas) — operação I/O localizada aqui
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}, Columns: {list(df.columns)[:10]}")

    # Detecta colunas relevantes e monta as transações em formato lista de listas
    group_col, item_col = detect_columns(df)
    print(f"Detected group column: {group_col}, item column: {item_col}")

    transactions = build_transactions(df, group_col, item_col)
    print(f"Built {len(transactions)} transactions (sample): {transactions[:3]}")

    if len(transactions) == 0:
        raise RuntimeError("No transactions could be built from CSV; adjust detection or input file")

    # Executa o algoritmo FP-Growth sobre as transações geradas
    print(f"Running fpgrowth minSupRatio={args.minsup} minConf={args.minconf} ...")
    freqItemSet, rules = fpgrowth(transactions, minSupRatio=args.minsup, minConf=args.minconf)

    print(f"Found {len(freqItemSet)} frequent itemsets and {len(rules)} rules")

    # Prepara payload com metadados e salva como pickle no caminho de saída
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.utcnow().isoformat(), "freq_itemsets": freqItemSet, "rules": rules}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
