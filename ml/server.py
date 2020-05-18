import torch
import flask
from flask import request
from flask_cors import CORS, cross_origin
import cv2
import os
from werkzeug.utils import secure_filename
from model import UNet

MODEL_PATH = "model/0.pth"
UPLOAD_PATH = "../backend/uploads/"

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location = "cpu"))

app = flask.Flask(__name__)
cors = CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_PATH
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods = ["GET", "POST"])
@cross_origin()
def upload_file():
    print("hi")
    if request.method == "POST":
        if "file" not in request.files:
            flask.flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file_name = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))

            inpt = cv2.imread("../backend/uploads/" + file_name)
            inpt = cv2.resize(inpt, (256, 256))

            image = torch.Tensor(2*(inpt/255) - 1).permute(2, 0, 1)

            image = model.generate(image)

            cv2.imwrite("../backend/outputs/" + "in_" + file_name, inpt)
            cv2.imwrite("../backend/outputs/" + "out_" + file_name, image)
            
            return flask.jsonify(filename = file_name)
    return "<h1>no</h1>"


@app.route("/grab_out", methods = ["GET"])
@cross_origin()
def grab_output():
    file_name = req.filename
    return send_from_directory("../backend/outputs/", "out_" + filename)

@app.route("/grab_in", methods = ["GET"])
@cross_origin()
def grab_input():
    file_name = req.filename
    return send_from_directory("../backend/outputs/", "in_" + filename)


app.run(host = "0.0.0.0", debug = True, port = 3000)
