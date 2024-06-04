from os.path import exists
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
import numpy as np


def predict():
    model_path = 'trained_model/mnist_predictor.tflite'

    if (not (exists(model_path))):
        print("Model not found\nCheck that 'mnist_predictor.tflite' model "
              "exists inside 'trained_model' directory.")
        return

    img_path = "./image.png"
    img = Image.open(img_path).convert('L').resize((28, 28),
                                                   Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32)
    res = Image.fromarray(img)

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = img

    input_tensor = img.reshape(input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    logits = interpreter.get_tensor(output_details[0]['index'])

    # Invert the `argsort` results, then get the top 2 largest values indices
    top_2_results = np.argsort(logits)[0][::-1][:2]

    return top_2_results


if __name__ == '__main__':
    prediction = predict()
    print("The top 2 most probable digits are: ")
    print(prediction)
