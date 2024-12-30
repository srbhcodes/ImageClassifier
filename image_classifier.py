from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pretrained model
model = MobileNetV2(weights="imagenet")

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    try:
        # Open and process the image
        image = Image.open(image_file)
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Make predictions
        predictions = model.predict(image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        return jsonify({
            'predictions': [
                {'label': label, 'probability': float(prob)}
                for (_, label, prob) in decoded_predictions
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
