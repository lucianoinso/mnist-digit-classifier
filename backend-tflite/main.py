import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from mnist_classifier import predict
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'yoursecretkey')
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def return_prediction():
    if request.method == 'POST':
        b64_image = request.form.to_dict(flat=False)['imageBase64'][0][22:]
        img_filename = 'image.png'
        with open(img_filename, "wb") as imagefile:
            imagefile.write(base64.b64decode(b64_image))
        prediction = predict(img_filename).astype(str).tolist()
        response = {'message': prediction}
    elif request.method == 'GET':
        response = {'message': 'Server running!'}
    else:
        response = {'message': "Method not allowed"}

    return jsonify(response)


if __name__ == '__main__':
    app.run()
