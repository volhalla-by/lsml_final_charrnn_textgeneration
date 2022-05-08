import os
import torch
from celery import Celery
from model import CharRNN, sample, load_mlflow_model

redis_host = os.environ.get('REDIS_HOST', 'localhost')
celery_app = Celery('tasks', backend=f'redis://{redis_host}', broker=f'redis://{redis_host}')

@celery_app.task
def celery_task(text):
    result = sample(model, size=500, prime=text, top_k=5)
    return result


if __name__ == '__main__':
    # model = torch.load('net_v2.model', map_location=torch.device('cpu'))

    # load model
    mlflow_url = f'http://mlflow:5003'
    experiment_name = 'char_rnn_books'
    model = load_mlflow_model(mlflow_url, experiment_name)
    model.eval()

    argv = [
        'worker',
        '--loglevel=INFO',
        '--pool=solo',
        '--concurrency=2'
    ]
    celery_app.worker_main(argv)
