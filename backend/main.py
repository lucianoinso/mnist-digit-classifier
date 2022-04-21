from flask import Flask, jsonify, request
from flask_cors import cross_origin, CORS
from mnist_classifier import predict
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yoursecretkey'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
def return_prediction():
    if request.method == 'POST':
        b64_image = request.form.to_dict(flat=False)['imageBase64'][0][22:]
        img_filename = 'image.png'
        with open(img_filename, "wb") as imagefile:
            imagefile.write(base64.b64decode(b64_image))
        prediction = predict(img_filename)
        response = {'message': prediction}
    else:
        response = {'message': "Method not allowed"}
    
    return jsonify(response)
