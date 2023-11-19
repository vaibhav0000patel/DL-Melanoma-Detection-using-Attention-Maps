# DL-Melanoma-Detection-using-Attention-Maps
Deep Learning based Melanoma Detection using Attention Maps

# Training the classifier

To set up and run the experiment, follow these instructions:

1. Initially, obtain the dataset, necessary for the experiment.

2. Compile all code cells, except for the final one.

2. In the last code cell, define the following parameters:

   - `data_dir`: Specify the path to your dataset.

   - `CNN`: Choose the neural network model for global and local feature extraction. Options include "8-layer", "ResNet18", and "ResNet50".

   - `ftr_size`: Determine the size of the local and global features to be extracted from the images.

   - `weight_sharing`: Set this to `True` if the same model weights are to be used across local features, global features, and the attention map extractor. Use `False` to employ separate models of the specified "CNN" type for each task.

   - `out_file`: Name the `.txt` file where the results of the experiment will be recorded. 


# UI Application (How to run)

## Introduction
This README provides instructions on how to set up and run the Flask application. Flask is a lightweight WSGI web application framework in Python, ideal for building small to medium-sized web apps.

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

## Running the Application
To run the Flask application, use the following command:
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000` in your web browser.
