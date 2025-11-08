import os
import pickle
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Set, Dict, Tuple

from flask import Flask, request, jsonify

VERSION = "v1.0.0"

app = Flask(__name__)

# Caminho padrão do modelo pickle (relativo à raiz do repositório)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "recommendation_model.pickle"

print(DEFAULT_MODEL_PATH)

# Permite sobrescrever via variável de ambiente MODEL_PATH
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Estado do modelo mantido na aplicação
app.model_rules = []  # tipo: ignore
app.model_mtime = None
app.model_lock = threading.Lock()


def load_model() -> None:
    """Carrega o modelo do disco e atualiza app.model_rules e app.model_mtime.

    Comentários em português: função segura para carregar o pickle do caminho
    configurado. Se o arquivo não existir, mantém rules vazias.
    """
    with app.model_lock:
        try:
            if not MODEL_PATH.exists():
                app.logger.warning(f"Modelo não encontrado em {MODEL_PATH}")
                app.model_rules = []
                app.model_mtime = None
                return

            mtime = MODEL_PATH.stat().st_mtime
            # se já carregado e sem mudança, nada a fazer
            if app.model_mtime is not None and app.model_mtime == mtime:
                return

            app.logger.info(f"Carregando modelo de {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as f:
                loaded = pickle.load(f)

            # O pickle pode ser uma lista de regras diretamente ou um dict com 'rules'
            if isinstance(loaded, dict) and "rules" in loaded:
                rules = loaded.get("rules", [])
            else:
                rules = loaded

            # normalizar: queremos uma lista de regras na forma [antecedent, consequent, confidence]
            app.model_rules = rules if isinstance(rules, list) else []
            app.model_mtime = mtime
            app.logger.info(f"Modelo carregado: {len(app.model_rules)} regras; mtime={datetime.fromtimestamp(mtime).isoformat()}")
        except Exception as e:
            app.logger.exception("Falha ao carregar modelo:")
            app.model_rules = []
            app.model_mtime = None


def reload_if_needed() -> None:
    """Recarrega o modelo caso o arquivo tenha sido modificado no disco."""
    try:
        if MODEL_PATH.exists():
            mtime = MODEL_PATH.stat().st_mtime
            if app.model_mtime is None or mtime != app.model_mtime:
                load_model()
        else:
            # Se o arquivo não existe e tínhamos um modelo carregado, limpar
            if app.model_mtime is not None:
                app.logger.warning("Modelo removido do disco; limpando modelo em memória")
                with app.model_lock:
                    app.model_rules = []
                    app.model_mtime = None
    except Exception:
        app.logger.exception("Erro verificando necessidade de reload do modelo")


# Carrega o modelo no momento da importação/startup quando possível.
# Importante: quando a imagem é executada com `flask run`, o bloco
# "if __name__ == '__main__'" não é executado, então chamamos
# explicitamente aqui para garantir que o modelo seja carregado
# no arranque do processo (se o arquivo existir).
try:
    load_model()
except Exception:
    # não falhar o processo se o modelo não puder ser carregado agora;
    # o modelo será recarregado sob demanda em `reload_if_needed()`.
    app.logger.exception("Falha ao carregar o modelo na inicialização")


def parse_rule(rule) -> Tuple[Set[str], Set[str], float]:
    """Extrai antecedente, consequente e confidence de uma regra em formatos variados.

    A fpgrowth_py tipicamente gera itens como sets ({'a'}) e confidence como float.
    Aqui convertemos para sets de strings e float.
    """
    try:
        if not isinstance(rule, (list, tuple)) or len(rule) < 3:
            return set(), set(), 0.0
        antecedent = set(map(lambda x: str(x).lower().strip(), rule[0]))
        consequent = set(map(lambda x: str(x).lower().strip(), rule[1]))
        confidence = float(rule[2])
        return antecedent, consequent, confidence
    except Exception:
        return set(), set(), 0.0


def recommend_from_rules(input_songs: List[str], top_k: int = 10) -> List[str]:
    """Gera recomendações simples a partir das regras carregadas.

    Estratégia:
    - normalizar músicas recebidas para lowercase
    - para cada regra (A -> B, conf), se A ⊆ input, pontuar cada item em B não presente em input
    - retornar top_k itens por soma das pontuações (confidence)
    """
    input_set = set([s.lower().strip() for s in input_songs if isinstance(s, str)])
    scores: Dict[str, float] = {}

    with app.model_lock:
        rules = list(app.model_rules) if app.model_rules else []

    for r in rules:
        antecedent, consequent, conf = parse_rule(r)
        if antecedent and antecedent.issubset(input_set):
            for item in consequent:
                if item not in input_set:
                    scores[item] = scores.get(item, 0.0) + conf

    # ordenar por score desc, depois alfabeticamente para estabilidade
    sorted_items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    recommendations = [item for item, _ in sorted_items][:top_k]
    return recommendations

@app.route("/")
def hello():
    return "API online!"

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Endpoint que recebe JSON {"songs": [...]} e retorna recomendações.

    Retorna JSON: {"songs": [...], "version": str, "model_date": str}
    """
    # Recarregar modelo se necessário antes de responder
    reload_if_needed()

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "request body is not valid JSON"}), 400

    if not payload or "songs" not in payload:
        return jsonify({"error": "JSON must contain 'songs' field with a list of song identifiers"}), 400

    print(payload)

    songs = payload.get("songs")
    if not isinstance(songs, list):
        return jsonify({"error": "'songs' must be a list"}), 400

    top_k_value = payload.get("top_k", 10)
    if isinstance(top_k_value, int):
        top_k = top_k_value
    elif isinstance(top_k_value, str) and top_k_value.isdigit():
        top_k = int(top_k_value)
    else:
        top_k = 10

    recs = recommend_from_rules(songs, top_k=top_k)

    model_date = datetime.fromtimestamp(app.model_mtime).isoformat() if app.model_mtime else None

    print(jsonify({"songs": recs, "version": VERSION, "model_date": model_date}))

    return jsonify({"songs": recs, "version": VERSION, "model_date": model_date})


if __name__ == "__main__":
    # Carrega o modelo na inicialização da aplicação
    load_model()
    # Comentário: executa o servidor Flask; altere PORT via variável de ambiente se necessário.
    app.run(debug=True, host="0.0.0.0", port=50024)
