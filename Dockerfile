FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances d'abord (profite du cache Docker si le code change sans que
# requirements.txt change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et les données nécessaires à l'entraînement du modèle
COPY src/ src/
COPY api/ api/
COPY data/ data/

# Entraîner et sérialiser le modèle au moment du build de l'image
# (le modèle .joblib est gitignored — on le régénère ici plutôt que d'en dépendre)
RUN python src/train.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
