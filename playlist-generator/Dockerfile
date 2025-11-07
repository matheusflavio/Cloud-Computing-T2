# Dockerfile para o container ML que gera o modelo de recomendação
FROM python:3.9-slim-bullseye

# Criar usuário não-root para segurança
RUN useradd --create-home appuser

WORKDIR /app

# Instala dependências primeiro (melhor cache de layers)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos Python necessários
COPY run_generator.py generator.py ./

# Define dono dos arquivos
RUN chown -R appuser:appuser /app

# Diretórios de dados/saída que serão montados como volumes
VOLUME ["/data", "/output"]

# Variáveis de ambiente úteis para executar o container
ENV DATA_PATH=/data/2023_spotify_ds1.csv
ENV OUTPUT_PATH=/output/recommendation_model.pickle

# Muda para usuário não-root
USER appuser

# Por padrão, executa o gerador lendo DATA_PATH e gravando OUTPUT_PATH
CMD ["python", "run_generator.py", "--csv", "/data/2023_spotify_ds1.csv", "--out", "/output/recommendation_model.pickle"]
