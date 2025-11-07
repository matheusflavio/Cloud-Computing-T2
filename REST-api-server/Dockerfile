# Dockerfile para o container da API REST
FROM python:3.9-slim-bullseye

# Criar usuário não-root para segurança
RUN useradd --create-home appuser

WORKDIR /app

# Instala dependências primeiro (melhor cache de layers)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY app.py ./

# Define dono dos arquivos
RUN chown -R appuser:appuser /app

# Diretório onde o modelo será montado via PVC
VOLUME ["/models"]

# Variáveis de ambiente padrão
ENV MODEL_PATH=/models/recommendation_model.pickle
ENV PORT=5000
ENV FLASK_APP=app.py
ENV WORKERS=2

# Expõe a porta do Flask/gunicorn
EXPOSE 5000

# Muda para usuário não-root
USER appuser

# Healthcheck para k8s (tenta carregar o modelo a cada 30s)
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c 'import pickle, os; pickle.load(open(os.environ["MODEL_PATH"], "rb"))' || exit 1

# Executa com gunicorn em produção
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--access-logfile", "-", "app:app"]