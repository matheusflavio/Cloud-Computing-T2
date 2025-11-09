import os
import pickle
from flask import Flask, request, jsonify
import time # Para checar a existência do arquivo

VERSION = "v1.0.0"
app = Flask(__name__)

# --- CORREÇÃO ---
# O deployment.yaml agora monta o volume em /recommend-rules
# Devemos ler o modelo de DENTRO desse volume
MODEL_PATH = "/recommend-rules/recommendation_model.pickle"
model_rules = None

def load_model():
    """Função para carregar o modelo do volume."""
    global model_rules
    if not os.path.exists(MODEL_PATH):
        print(f"Aguardando modelo em: {MODEL_PATH}...")
        return False
    
    try:
        with open(MODEL_PATH, 'rb') as file:
            model_rules = pickle.load(file)
        print(f"Modelo carregado com sucesso de: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return False

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

# Tenta carregar o modelo na inicialização
load_model()

@app.route("/")
def hello():
    if model_rules:
        return f"API online! Modelo carregado. Versão: {VERSION}"
    else:
        return f"API online! ATENÇÃO: Modelo NÃO carregado. Verificando {MODEL_PATH}. Versão: {VERSION}"


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Endpoint que recebe JSON {"songs": [...]} e retorna recomendações."""
    global model_rules
    
    # Se o modelo não carregou na inicialização, tenta de novo.
    # Isso ajuda caso a API tenha iniciado antes do Job terminar.
    if not model_rules:
        print("Modelo não estava carregado. Tentando recarregar...")
        if not load_model():
             return jsonify({"error": "Modelo ainda não está pronto. O Job pode estar em execução."}), 503 # Service Unavailable

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

    model_rules = '../ml_rules/model_rules.pkl'
    
    with open(model_rules, 'rb') as rules:
        music_rules = pickle.load(rules)

    # Obter recomendações
    recs = recommend_from_rules(songs, music_rules, top_k)

    # Tenta pegar a data de modificação do arquivo do modelo
    try:
        model_time = time.ctime(os.path.getmtime(MODEL_PATH))
    except Exception:
        model_time = "desconhecida"

    return jsonify({"songs": recs, "version": VERSION, "model_date": model_time})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=50024)