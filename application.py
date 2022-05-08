from flask import Flask, request, render_template
from celery.result import AsyncResult
from tasks import celery_app, celery_task


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        task = celery_task.delay(text)
        task_id = task.id
        # result = sample(model, size=100, prime=text, top_k=6)
        response = {'task_id': task_id}
        return response

    return None


@app.route('/task/<task_id>', methods=['GET'])
def task(task_id):
    task = AsyncResult(task_id, app=celery_app)
    return {
        'ready': task.ready(),
        'result': str(task.result) if task.ready() else None
    }


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
