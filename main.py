import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

def load_image_opencv(original):
    original = cv2.imread(original)
    resized = cv2.resize(original, (150, 150))
    dataXG = np.array(resized) / 255.0
    return dataXG

model = load_model('testgcp_model.h5')

# {'Chickenpox': 0, 'Measles': 1, 'Monkeypox': 2, 'Normal_image': 3}
def decode_labels(score):
    if score==0:
        return 'Checkenpox'
    elif score==1:
        return 'Measles'
    elif score==2:
        return 'Monkeypox'
    else:
     return 'Normal_image'

# load image
# img = load_image_opencv('aug_0_2751.png')

# # label
# label = prediction(img)
# print(label)

app = Flask(__name__)

UPLOAD_FOLDER = '/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST" or request.method == 'GET':
        # file = request.files.get('file')
        # file=request.form.values('filename')
        file = request.files["filename"]
        if file is None or file.filename == "":
            return jsonify('error', 'no files')
        else:
            try:
                # else:
                #f="//ad.monash.edu/home/User066/csit0004/Desktop/Monkeypox-dataset-2022/Monkeypox-dataset/Augmented_/f2/val/Measles/measles_aug_0_4201.png"
                # filename = secure_filename(file.filename)
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                original = cv2.imread(file.filename)
                resized = cv2.resize(original, (150, 150))  # error occurred
                dataXG = np.array(resized) / 255.0
                # print(dataXG)
                # reshape
                img = np.expand_dims(dataXG, axis=0)
                predictions = model(img)
                # # predictions=tf.nn.softmax(predictions)
                pred0 = predictions[0]
                label0 = np.argmax(pred0)
                #data = {'prediction': int(label0)}
                data=decode_labels(int(label0))
                return render_template('index.html', prediction_text=data)

            except Exception as e:
                return jsonify('error', str(e))


if __name__ == '__main__':
    app.run(debug=True)
