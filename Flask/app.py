"""
code : swati sinha & maitry sinha
"""
from flask import Flask, render_template, request
from flask_cors import cross_origin
from tensorflow.keras.models import load_model
import cv2
from smooth_blending import Final_prediction
model = load_model('model/satellite_256_unet_model_c6.hdf5', compile=False)


from utils import Decode


app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
@cross_origin()
def result():
    if request.method == 'POST':
        image = request.json['image']
        img = Decode(image).copy()
        ht, wd = img.shape[:2]
        img = cv2.resize(img, (510, 510*ht//wd), interpolation=cv2.INTER_NEAREST)
        Final_prediction(img, model, patch_size=256, n_classes=6)
        return render_template('index.html')
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
