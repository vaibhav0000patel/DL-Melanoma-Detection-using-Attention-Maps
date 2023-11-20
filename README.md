# DL-Melanoma-Detection-using-Attention-Maps
Deep Learning based Melanoma Detection using Attention Maps

# Training the classifier

To set up and run the experiment, follow these instructions:

1. Initially, obtain the dataset using `01DataPreprocessing/skin_lesion_imgs_augmentation.ipynb`, necessary for the experiment.

2. Compile all code cells in `02MelanomaClassification/Melanoma_detection_main_code.ipynb`, except for the final one.

3. In the last code cell, define the following parameters:

   - `data_dir`: Specify the path to your dataset.

   - `CNN`: Choose the neural network model for global and local feature extraction. Options include "8-layer", "ResNet18", and "ResNet50".

   - `ftr_size`: Determine the size of the local and global features to be extracted from the images.

   - `weight_sharing`: Set this to `True` if the same model weights are to be used across local features, global features, and the attention map extractor. Use `False` to employ separate models of the specified "CNN" type for each task.

   - `out_file`: Name the `.txt` file where the results of the experiment will be recorded. 

4. Run the last cell for training. The trained model will be saved in your Google Drive as "melanoma_detection.pt". Then, use your trained model in the UI Application, as shown below.

# UI Application (How to run)

## Prerequisites
Before you start, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

## Installation

### Clone the Repository
First, clone the repo to your local machine using Git:
```bash
git clone [URL of the Git repository]
cd [repository name]
```

### Setting Up a Virtual Environment
It's recommended to run Python projects in a virtual environment to manage dependencies. To set up a virtual environment, run:
```bash
python -m venv venv
```
Activate the virtual environment:
- On Windows: `venv\Scripts\activate`
- On Unix or MacOS: `source venv/bin/activate`

### Install Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### Use your trained model
1. Move your model to `03UIApplication\static\models`
2. Set the name of your model in `app.py` => `MODEL_PATH = "static/models/<your-model>.pt"`

## Running the Application
To run the Flask application, use the following command:
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000` in your web browser.
