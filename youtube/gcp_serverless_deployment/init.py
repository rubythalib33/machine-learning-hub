from transformers import pipeline

classifier = pipeline("text-classification", model = "Souvikcmsa/BERT_sentiment_analysis")

#save the model
classifier.save_pretrained("model_sentiment")