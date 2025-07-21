import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
import tensorflow as tf
import numpy as np
import cv2
import dlib
import pickle
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the model
def get_model():
    backbone = tf.keras.applications.EfficientNetB2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

model = get_model()
model.load_weights("FacialExpressionModel.keras")

# Load LabelEncoder
def load_object(name):
    with open(f"{name}.pck", "rb") as f:
        return pickle.load(f)

Le = load_object("LabelEncoder")

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

def ProcessImage(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [96, 96], method="bilinear")
    image = tf.expand_dims(image, 0)
    return image

def RealtimePrediction(image, model, encoder_):
    # Set verbose=0 to prevent prediction progress bars in the server log
    prediction = model.predict(image, verbose=0)
    prediction = np.argmax(prediction, axis=1)
    return encoder_.inverse_transform(prediction)[0]

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_array = np.array(img)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 0)

        if len(rects) >= 1:
            # Process the first detected face, matching the notebook logic
            rect = rects[0]
            (x, y, w, h) = rect_to_bb(rect)
            height, width = gray.shape

            # Add a 10px margin around the face, same as the notebook
            y1 = max(y - 10, 0)
            y2 = min(y + h + 10, height)
            x1 = max(x - 10, 0)
            x2 = min(x + w + 10, width)
            
            # --- START OF CORRECTION (to match notebook) ---
            # 1. Crop the face from the GRAYSCALE image
            face_img = gray[y1:y2, x1:x2]

            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                return jsonify({'error': 'Invalid face region'})

            # 2. Convert the grayscale face crop back to a 3-channel RGB image
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            
            # 3. Process the 3-channel grayscale image for the model
            processed = ProcessImage(face_rgb)
            # --- END OF CORRECTION ---

            prediction = RealtimePrediction(processed, model, Le)

            return jsonify({'prediction': prediction, 'bbox': [x, y, w, h]})
        else:
            return jsonify({'error': 'No face detected'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)