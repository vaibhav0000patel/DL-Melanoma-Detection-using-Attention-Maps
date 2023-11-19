#! /usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from attentionnet import AttentionNetwork
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
from torchvision.utils import save_image


STATIC_FOLDER = 'static/'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = STATIC_FOLDER+UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_URL'] = "/"+UPLOAD_FOLDER
app.secret_key = "supersecretkey"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure you're using the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model (replace with your model's path)

ftr_size = 512

CNN_model1 = models.resnet50(pretrained=True)
fc_in_ftrs = CNN_model1.fc.in_features
CNN_model1.fc = nn.Linear(fc_in_ftrs, ftr_size)

CNN_model2 = models.resnet50(pretrained=True)
CNN_model2.fc = nn.Linear(fc_in_ftrs, ftr_size)

CNN_model3 = models.resnet50(pretrained=True)
CNN_model3.fc = nn.Linear(fc_in_ftrs, ftr_size)

attention_net = CNN_model1
attention_ftr_extractor = CNN_model2
global_net = CNN_model3

model = AttentionNetwork(AttentionNet=attention_net,
                           AttentionFtrExtractor=attention_ftr_extractor,
                           GlobalNet=global_net,
                           num_classes=3,
                           size=224,
                           ftr_size=ftr_size)

model.load_state_dict(torch.load('static/models/model_epoch_11.pt')['state_dict'])

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_path_url = os.path.join(app.config['UPLOAD_FOLDER_URL'], filename)
            file.save(file_path)
            
            image = Image.open(file_path)
            image = transform(image).unsqueeze(0).to(device)
            
            prediction = [0,0,0]

            with torch.no_grad():
                outputs, img_with_rect = model(image)
                save_image(img_with_rect,file_path)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                prediction = probs.cpu().numpy().tolist()[0]
            
            prediction = list(map(lambda x:round(x,4)*100,prediction))

            return render_template('index.html', prediction=prediction, image_path=file_path_url, filename=filename)
    
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
