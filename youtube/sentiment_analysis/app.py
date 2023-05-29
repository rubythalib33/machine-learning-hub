from engine import sentiment_analysis, translate
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def home():
    return "Sentiment Analysis API"

@app.post("/predict")
def predict(text: str, language: str = "id"):
    text_en = translate(text, language)
    label, score = sentiment_analysis(text_en)
    return {"label": label, "score": score, "text": text_en}

# Path: youtube/sentiment_analysis/requirements.txt
