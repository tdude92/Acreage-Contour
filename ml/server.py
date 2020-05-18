import torch
import flask
from flask import request, redirect, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import io
import cv2
import os
from werkzeug.utils import secure_filename
import base64
from model import UNet
import json
from PIL import Image
import re

MODEL_PATH = "model/0.pth"
UPLOAD_PATH = "../backend/uploads/"

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

app = flask.Flask(__name__)
app.secret_key = "super secret key"
cors = CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_PATH
app.config['CORS_HEADERS'] = 'Content-Type'

ctr = 0


@app.route("/", methods=["GET", "POST"])
@cross_origin()
def upload_file():
    global ctr
    if request.method == "POST":
        ctr += 1
        b64_img = request.get_json()["data"]
        b64_img = re.sub('^data:image/.+;base64,', '', b64_img)

        inpt = np.array(Image.open(io.BytesIO(base64.b64decode(b64_img))))
        inpt = cv2.resize(inpt, (256, 256))

        image = torch.Tensor(2*(inpt/255) - 1).permute(2, 0, 1)

        image = model.generate(image)

        inpt = cv2.cvtColor(inpt.astype("float32"), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image.astype("float32"), cv2.COLOR_RGB2BGR)

        retval, inpt = cv2.imencode(".jpg", inpt)
        retval, image = cv2.imencode(".jpg", image)

        inpt = base64.b64encode(inpt)
        image = base64.b64encode(image)

        inpt = str(inpt)[2:]
        image = str(image)[2:]

        inpt = inpt[:len(inpt) - 1]
        image = image[:len(image) - 1]

        return flask.jsonify(input=inpt, output=image)
    return "<h1>no</h1>"


app.run(host="localhost", debug=True, port=5000)
