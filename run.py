import pickle
from tensorflow import keras
import librosa
from flask import *
import numpy as np
import subprocess


outfile = open('model/scores.json', "r")
scores = json.load(outfile)
max_solution = max(scores, key=scores.get)

if max_solution == "DeepLearning":
    model = keras.models.load_model('model/speech2text_model')
    model.summary()
else:
    with open('model/speech2text_model.sav', "rb") as f:
        model = pickle.load(f)

label_map = {"Excel": 0, "Word": 1, "Google": 2, "Note": 3, "PP": 4}
app_map = {"Excel": "C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
           "Word": "C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
           "Google": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
           "Note": "C:\Windows\system32\\notepad.exe",
           "PP": "C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE"}
app = Flask(__name__)
@app.route('/')
def main():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['audio_data']
        samples, sample_rate = librosa.load(f, sr=8000)
        samples = np.abs(librosa.stft(samples))
        samples = samples[:, :50]
        if samples.shape[1] < 50:
            shape_inves = 50 - samples.shape[1]
            samples = np.concatenate((samples, np.array([[0]*shape_inves] * 1025)), 1)
        samples = np.array([samples])
        if max_solution == "DeepLearning":
            samples = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], 1)
            output_predict = np.argmax(model.predict(samples)[0])
        else:
            samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])
            output_predict = model.predict(samples)[0]
        output_string = [k for k, v in label_map.items() if v == output_predict][0]
        print("Open: ", output_string)
        subprocess.call(app_map[output_string])

        return output_string
        # return render_template("index.html", name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)
