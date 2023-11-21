#! /usr/bin/env python3

import os
from attentionnet import Model
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Constants
STATIC_FOLDER = 'static/'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Model to use
MODEL_PATH = "static/models/model_epoch_43.pt"

# Flask configurations
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = STATIC_FOLDER+UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_URL'] = "/"+UPLOAD_FOLDER
app.secret_key = "supersecretkey"

# Load the model
model = Model(MODEL_PATH)

# Allowed file types that the user can upload
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# The following functions is called on form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # Check if the form data has image
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        # Pick the uploaded image
        file = request.files['file']

        # Check if there is filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Check if the file has valid extention
        if file and allowed_file(file.filename):
            
            # First save the file locally
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_path_url = os.path.join(app.config['UPLOAD_FOLDER_URL'], filename)
            file.save(file_path)
            
            # Then use the model to find the classification predictions
            prediction = model.predict(file_path)

            # Return the predictions, name of the file and image file with attention map
            return render_template('index.html', prediction=prediction, image_path=file_path_url, filename=filename)
    
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
