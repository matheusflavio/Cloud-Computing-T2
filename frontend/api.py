from flask import Flask, request, jsonify
import pickle
import os


app = Flask(__name__)


def playlist_recommender(musics, rules):
    
    playlist = set()

    for song in musics:
        consequents = set()
        
        # Filtra as regras onde o antecedente contém a música em questão
        filter_rules = rules[rules['antecedents'].apply(lambda x: song in x)]
        
        # Para cada consequent da regra filtrada, adiciona na lista de consequentes
        for consequent in filter_rules['consequents']:
            consequents.update(consequent)
        
        # Convertendo o conjunto de consequentes para lista
        consequents = list(consequents)
        
        contador = 0
        
        # Para cada música nos consequentes, adiciona na playlist (sem repetir)
        for music in consequents:
            if contador == 5:
                break
            if music not in playlist:
                playlist.add(music)
                contador += 1

    if len(playlist) < 5:
        # Filtra músicas que não estão na playlist e que não foram ouvidas
        remaining_musics = set(rules['consequents'].explode()).difference(playlist).difference(musics)
        remaining_to_add = 5 - len(playlist)
        playlist.update(list(remaining_musics)[:remaining_to_add])

    return list(playlist)




@app.route("/")
def hello():
	return "API working now!"


@app.route('/api/recommend/', methods=["POST"])
def api_music():
    req = request.get_json(force=True)
    print(req['songs'])
    
    pkl_ = '../ml_rules/model_rules.pkl'
    
    with open(pkl_, 'rb') as rules:
        music_rules = pickle.load(rules)
        
    list_musics = playlist_recommender(req['songs'], music_rules)

    
    return jsonify({'songs': list_musics, 'version': 1, 'model_date': "date"})	


if __name__ == '__main__':
	app.run(debug=True,host='0.0.0.0', port=50024)