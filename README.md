Sure! Here’s a detailed document that explains the entire process of setting up and running the image classification API using Flask and TensorFlow. This document includes every step, from setting up the environment to running the Flask server and testing the API.

---

# **Document: Achieving Image Classification with Flask and TensorFlow**

## **1. Introduction**

This project involves creating a simple image classifier API using **Flask** and **TensorFlow**. The Flask application serves as the backend to handle image uploads, classify images using a pre-trained **MobileNetV2** model, and return predictions in JSON format. The project was implemented step by step, and the following guide outlines each stage.

---

## **2. Setting Up the Environment**

### **2.1 Install Python**

- Python was installed on the system to ensure compatibility with the Flask and TensorFlow libraries.
  - Verify the Python installation with the command:
    ```bash
    python --version
    ```

### **2.2 Install Visual Studio Code (VS Code)**

- Visual Studio Code was installed as the Integrated Development Environment (IDE) to write, edit, and manage the project files.

---

## **3. Creating the Project Directory**

### **3.1 Create a New Directory for the Project**

- We created a new folder named `ai_image_classifier` to hold all project files.
  - In VS Code, open the terminal and navigate to the project directory.
  - The folder will contain:
    - `image_classifier.py` (the main script for Flask API and image classification logic).

---

## **4. Setting Up the Virtual Environment**

### **4.1 Create a Virtual Environment**

- A virtual environment was created to isolate the project’s dependencies.
  - The following command was used to create the virtual environment:
    ```bash
    python -m venv env
    ```

### **4.2 Activate the Virtual Environment**

- We activated the virtual environment to ensure the dependencies were installed in the isolated environment:
  - For Linux/Mac:
    ```bash
    source env/bin/activate
    ```
  - For Windows:
    ```bash
    .\env\Scripts\activate
    ```

---

## **5. Installing Required Libraries**

### **5.1 Install Flask and TensorFlow**

- We installed the required libraries: **Flask**, **TensorFlow**, and **Pillow** (for image processing).
  - The following command was used to install all dependencies:
    ```bash
    pip install tensorflow flask pillow
    ```

---

## **6. Writing the Flask Application Code**

### **6.1 Setting Up the Flask API**

- We wrote the Flask code to set up an API that listens for `POST` requests at the `/classify` endpoint. The code performs the following tasks:
  - Accepts image files uploaded via `POST` requests.
  - Preprocesses the image (resize and normalize) for classification.
  - Uses a **MobileNetV2** pre-trained model from TensorFlow to classify the image.
  - Decodes the prediction results and returns the top 3 predicted labels with their probabilities in JSON format.

```python
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
        # Open the image file
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
    app.run(debug=True)
```

### **6.2 Explanation of the Code**:
- **Model Loading**: We load the pre-trained **MobileNetV2** model from TensorFlow, which is trained on the **ImageNet** dataset.
- **Image Preprocessing**: When an image is uploaded, we:
  1. Resize it to **224x224** pixels (the size expected by MobileNetV2).
  2. Convert the image into an array format suitable for TensorFlow.
  3. Preprocess the image using `preprocess_input` to ensure it's in the correct format for prediction.
- **Prediction and Decoding**: The image is passed through the model, and the top 3 predicted labels are decoded using the `decode_predictions` method.
- **JSON Response**: The predictions (labels and probabilities) are returned as a JSON response.

---

## **7. Running the Flask Application**

### **7.1 Start the Flask Server**

- We ran the Flask application using the command:
  ```bash
  python image_classifier.py
  ```
- The application started running on `http://127.0.0.1:5000` (local) and was also accessible via the network at `http://172.21.59.110:5000`.

### **7.2 Application Status**

- The Flask app was in **debug mode**, meaning it would automatically restart when code changes were detected.
- The server would listen to incoming requests and serve the `/classify` endpoint.

---

## **8. Testing the Image Classification API**

### **8.1 Using cURL to Send Image to the Server**

- We used **cURL** to send an image file to the Flask API for classification:
  ```bash
  curl -X POST -F "image=@image.jpg" http://172.21.59.110:5000/classify
  ```

- The server responded with predictions in the following format:
  ```json
  {
    "predictions": [
      {
        "label": "daisy",
        "probability": 0.1095186248421669
      },
      {
        "label": "bee",
        "probability": 0.0955282524228096
      },
      {
        "label": "rapeseed",
        "probability": 0.050430405884981155
      }
    ]
  }
  ```

### **8.2 Testing the Application in the Browser**

- While the API worked with cURL, testing via the browser led to the error **"Method Not Allowed"**.
  - This is because browsers use **GET** requests by default, while the `/classify` endpoint accepts **POST** requests.
- The solution was to test the API with **POST** requests (e.g., using **Postman** or **cURL**), not directly in the browser.

---

## **9. Debugging and Resolving Issues**

### **9.1 Handling Errors and Debugging the Application**

- During the setup, we encountered several errors:
  1. **No Image Uploaded**: We handled the case where no image was provided in the request using a condition to check for the image field.
  2. **Internal Server Errors**: Errors related to image processing or model prediction were caught and returned in the response with appropriate error messages.
  
---

## **10. Conclusion**

We successfully built and tested a simple Flask-based image classification API using **TensorFlow**'s MobileNetV2 model. The API can classify images and return top predictions with probabilities. This solution leverages the power of pre-trained models and allows easy deployment for image classification tasks.

### **Next Steps**:
- **Frontend Integration**: The backend API can be integrated into a frontend application (e.g., React) to make the classification service more user-friendly with a file upload interface.
- **Deployment**: The Flask app can be deployed on cloud platforms like **Heroku** or **AWS** to make it accessible globally.

---

This is a summary of the entire process from setup to implementation. Let me know if you need further assistance or want to move on to the next stage, such as frontend integration or deployment!
