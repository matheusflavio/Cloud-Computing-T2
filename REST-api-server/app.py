import os
import pickle
from flask import Flask, request, jsonify

VERSION = "v1.0.0"
app = Flask(__name__)

def recommend_from_rules(input_songs, model_rules, top_k=10):
    """Gera recomendações simples com base nas regras"""
    input_set = set(map(lambda s: s.lower().strip(), input_songs))

    # Filtrando as regras de forma simples
    recommended = set()
    for rule in model_rules:
        antecedent, consequent, conf = rule
        if set(map(lambda x: x.lower().strip(), antecedent)).issubset(input_set):
            recommended.update(map(lambda x: x.lower().strip(), consequent))

    # Limitar ao top_k
    return list(recommended)[:top_k]


@app.route("/")
def hello():
    return "API online!"


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Endpoint que recebe JSON {"songs": [...]} e retorna recomendações."""
    # Carregar o modelo (pode ser carregado na hora, de forma simplificada)
    req = request.get_json(force=True)

    model_rules = 'recommendation_model.pickle'

    with open(model_rules, 'rb') as rules:
        model_rules = pickle.load(rules)

    if not model_rules:
        return jsonify({"error": "Modelo não encontrado ou falha ao carregar."}), 500

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body não é JSON válido"}), 400

    if not payload or "songs" not in payload:
        return jsonify({"error": "JSON precisa conter o campo 'songs' com uma lista de músicas."}), 400

    songs = payload.get("songs")
    if not isinstance(songs, list):
        return jsonify({"error": "'songs' deve ser uma lista"}), 400

    top_k_value = payload.get("top_k", 10)
    top_k = int(top_k_value) if isinstance(top_k_value, int) or (isinstance(top_k_value, str) and top_k_value.isdigit()) else 10

    # Obter recomendações
    recs = recommend_from_rules(songs, model_rules, top_k)

    return jsonify({"songs": recs, "version": VERSION, "model_date": "data do modelo (não implementado)"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=50024)