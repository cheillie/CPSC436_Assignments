FROM public.ecr.aws/lambda/python:3.8

COPY sentiment_analysis_lambda.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY nltk_data /nltk_data

CMD [ "sentiment_analysis_lambda.lambda_handler" ]
