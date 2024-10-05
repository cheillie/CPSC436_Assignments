import os
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.data.path.append("/nltk_data")

def lambda_handler(event, context):
    def sentiment_analysis(text):
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)

        if sentiment_scores['compound'] > 0:
            response = "Positive sentiment"
        elif sentiment_scores['compound'] < 0:
            response = "Negative sentiment"
        else:
            response = "Neutral sentiment"
        return response
    result = sentiment_analysis(event['text'])
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }