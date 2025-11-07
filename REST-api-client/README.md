# REST API Client

Pequeno cliente CLI para testar o endpoint `/api/recommend` do servidor de recomendação.

Instalação mínima (recomenda-se criar um virtualenv):

```powershell
python -m pip install -r REST-api-client/requirements.txt
```

Exemplos de uso:

- Enviar músicas diretamente:

```powershell
python REST-api-client/client.py --songs "Yesterday" "Bohemian Rhapsody"
```

- Amostrar N músicas do dataset e enviar:

```powershell
python REST-api-client/client.py --from-csv ..\datasets\2023_spotify_songs.csv --count 2
```

O cliente grava a resposta HTTP completa no ficheiro `response.out` e também imprime o JSON resumido no terminal.

Observações:
- O endpoint alvo padrão é `http://localhost:30502` e o path usado é `/api/recommend`.
- Se o seu servidor usar outra porta, passe `--host http://localhost:<PORT>`.
