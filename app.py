import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow import keras
#from tensorflow.keras.models import load_model
#from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/select.html', )

@app.route('/predict', methods=['POST'])
def predict():
    chosen_model = request.form['select_model']
    model_dict = {'CNN'   :   r'G:\UMM\Semester 7\ML\Cardiomegaly-Disease-Classification-CNN\static\MLModule\model cnn 6.h5',
                  'Transferlearning'     :   r'G:\UMM\Semester 7\ML\Cardiomegaly-Disease-Classification-CNN\static\MLModule\model transfer learning 6.h5',}
    if chosen_model in model_dict:
        model = keras.models.load_model(model_dict[chosen_model]) 
    else:
        model = keras.models.load_model(model_dict[0])
    file = request.files["file"]
    file.save(os.path.join(r'G:\UMM\Semester 7\ML\Cardiomegaly-Disease-Classification-CNN\static', 'temp.jpg'))
    # img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    # img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    c = []
    image = cv2.imread(r'G:\UMM\Semester 7\ML\Cardiomegaly-Disease-Classification-CNN\static\temp.jpg')
    image = cv2.resize(image, (128,128))
    c.append(image)
    c = np.array(c)
    start = time.time()
    if chosen_model == 'CNN' :
        pred = model.predict(c)
        labels = (pred > 0.5).astype(np.int)
    else :
        pred = model.predict([c,c])
        labels = []
        if pred[0][0]>pred[0][1]:
            labels.append(0)
            # n = pred[0][0]
        else:
            labels.append(1)
            # n = pred[0][1]
    print(labels)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred[0]]
    return predict_result(chosen_model, runtimes, respon_model, 'temp.jpg')

def predict_result(model, run_time, probs, img):
    class_list = {'False': 0, 'True': 1}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)

if __name__ == "__main__": 
        app.run(debug=True, host='0.0.0.0', port=2000)
