import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import requests


def detect_title_column(df_columns: List[str]) -> Optional[str]:
    """Detecta automaticamente qual coluna do CSV provavelmente contém o título da música."""
    candidates = [
        "track_name",
        "track",
        "name",
        "title",
        "song",
        "song_name",
    ]
    cols_lower = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def sample_songs_from_csv(csv_path: Path, count: int) -> List[str]:
    """Carrega o CSV de músicas e devolve uma lista com `count` títulos escolhidos aleatoriamente.

    Implementação leve sem depender de pandas (usa split/strip). Se o arquivo for grande,
    uma implementação com csv.reader seria mais robusta. Aqui priorizamos simplicidade.
    """
    import csv

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # lê cabeçalho para detectar coluna
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []

        title_col = detect_title_column(header)
        if title_col is None:
            # fallback: use the first column
            title_col = header[0]
        title_idx = header.index(title_col)

        songs = []
        for row in reader:
            if len(row) > title_idx:
                val = row[title_idx].strip()
                if val:
                    songs.append(val)

    if not songs:
        return []

    if count >= len(songs):
        return songs

    return random.sample(songs, count)


def send_request(host: str, songs: List[str], top_k: int, out_file: Path) -> None:
    url = host.rstrip("/") + "/api/recommend"
    payload = {"songs": songs, "top_k": top_k}
    headers = {"Content-Type": "application/json"}

    print(f"Enviando {len(songs)} músicas para {url} (top_k={top_k})")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        print(f"Erro ao conectar-se ao servidor: {e}")
        return

    # grava resposta inteira em response.out para inspeção
    try:
        with out_file.open("w", encoding="utf-8") as f:
            f.write(f"HTTP {resp.status_code}\n")
            f.write(json.dumps(resp.headers, default=str, indent=2))
            f.write("\n\n")
            # tenta mostrar JSON bonito quando possível
            try:
                body = resp.json()
                f.write(json.dumps(body, indent=2, ensure_ascii=False))
            except Exception:
                f.write(resp.text)

        print(f"Resposta gravada em {out_file} (status {resp.status_code})")
        try:
            # também imprime o JSON resumido no terminal quando disponível
            print("Resposta (JSON):")
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(resp.text[:1000])
    except Exception as e:
        print(f"Falha ao gravar resposta: {e}")


def main():
    parser = argparse.ArgumentParser(description="REST API client for playlist recommender")
    parser.add_argument("--host", default="http://localhost:30502", help="Server base URL (default http://localhost:30502)")
    parser.add_argument("--songs", nargs="*", help="List of song titles to send (quoted if contain spaces)")
    parser.add_argument("--from-csv", dest="csv", help="Path to CSV with songs (e.g. ../datasets/2023_spotify_songs.csv)")
    parser.add_argument("--count", type=int, default=1, help="Number of random songs to sample from CSV when --from-csv is used")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations requested")
    parser.add_argument("--out", default="response.out", help="Output file to save server response")
    args = parser.parse_args()

    songs: List[str] = []
    if args.songs:
        songs = args.songs
    elif args.csv:
        csv_path = Path(args.csv)
        try:
            songs = sample_songs_from_csv(csv_path, args.count)
            if not songs:
                print("Nenhuma música encontrada no CSV especificado.")
                return
            print(f"Amostradas {len(songs)} músicas do CSV {csv_path}")
        except Exception as e:
            print(f"Erro lendo CSV: {e}")
            return
    else:
        parser.print_help()
        return

    send_request(args.host, songs, args.top_k, Path(args.out))


if __name__ == "__main__":
    main()
