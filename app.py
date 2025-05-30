from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('model.h5')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load class labels, aliases, and remedies
try:
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    with open("remedy_dict.json", "r") as f:
        remedy_dict = json.load(f)
    with open("class_aliases.json", "r") as f:
        class_aliases = json.load(f)
except Exception as e:
    class_labels = []
    remedy_dict = {}
    class_aliases = {}
    print(f"Error loading JSON data: {e}")

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None, remedy_text=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Delete previous uploads
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return render_template('index.html', prediction_text='No file selected.')

        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        predicted_disease = class_aliases.get(predicted_class, predicted_class)
        remedy_text = remedy_dict.get(predicted_class, "No remedy information available.")
        
        # Decide whether to show disease/remedy
        show_remedy = not predicted_disease.lower().startswith("it's healthy")

        prediction_text =f" Predicted Disease: {predicted_disease}"  if show_remedy else "The plant is healthy! 🌿"
        remedy_text = remedy_text if show_remedy else ""

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            remedy_text=remedy_text,
            image_path=filepath,
            show_remedy=show_remedy
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
