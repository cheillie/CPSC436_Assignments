FROM public.ecr.aws/lambda/python:3.8

COPY image_classification_lambda.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "image_classification_lambda.lambda_handler" ]

