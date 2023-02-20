from flask import Flask, render_template, request
from inference import inference

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(filename)

        machine_type = request.values['machine_type']
        machine_id = request.values['machine_id']

        anomaly_score = inference(filename, machine_type, machine_id)
        msg = "异常得分：{}".format(anomaly_score)
        return msg
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
