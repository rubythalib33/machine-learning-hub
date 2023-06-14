from transformers import pipeline
import requests

classifier = pipeline("text-classification", model = "model_sentiment")
# run the classifier in cpu
classifier.model.to('cpu')


def sentiment_analysis(text):
    result = classifier(text)[0]
    return result['label'], result['score']


def translate(text, lang_source="id"):
    lang_target = "en"
    translate_api = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={lang_source}&tl={lang_target}&dt=t&q={text}"
    response = requests.get(translate_api)
    if response.status_code == 200:
        return response.json()[0][0][0]
    else:
        return "Error"