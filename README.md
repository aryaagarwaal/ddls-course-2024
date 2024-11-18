# ddls-course-2024
DDLS Project
Identification of Metastatic and Primary Melanoma Tumor WSI using Vision Transformer (ViT)
This repository contains a Python script for the identification of metastatic and primary melanoma tumor whole-slide images (WSI). The script processes WSI files, extracts regions of interest (ROIs), trains a Vision Transformer (ViT) model, and visualizes model predictions using Grad-CAM.

# Installation
  Ensure you have the necessary dependencies installed. 
  !pip install openslide-python
  !apt-get install openslide-tools
  !pip install Pillow
  !pip install tensorflow scikit-learn matplotlib opencv-python scikit-image

# Google Drive Setup
  The script assumes that your data is stored in Google Drive. 
  from google.colab import drive
  drive.mount('/content/drive')

# Import Required Libraries
  Libraries for image processing, machine learning, and file management are imported.

# Path Specification
  Update the paths to your data:
  primary_folder: Directory for primary tumor WSIs.
  metastatic_folder: Directory for metastatic tumor WSIs.
  output_folder: Folder to store the processed and downsampled images.

# Image Processing
  The script downsamples .svs images and extracts regions of interest (ROIs) based on color segmentation. It then saves these ROIs as .png files.
  process_svs_file(primary_folder, output_folder, level=1)
  process_svs_file(metastatic_folder, output_folder, level=1)

# ROI Extraction
  Regions of interest are extracted from both primary and metastatic WSIs. These are resized to 224x224 pixels, making them ready for input to the machine-learning model.
  process_folder(primary_folder, primary_save_folder, 0)
  process_folder(metastatic_folder, metastatic_save_folder, 1)

# Dataset Preparation
  The script loads images and labels from the folders, shuffles the data, and splits it into training and testing sets (80% train, 20% test).
  X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42). The dataset is balanced using class weights and data augmentation functions

# Model Training
  The Vision Transformer (ViT) model is used to classify the tumor images. The model is trained for 25 epochs on the training dataset.
  vit_model.fit(train_dataset, validation_data=test_dataset, epochs=25)

# Evaluation
  After training, the script evaluates the model by calculating accuracy, precision, recall, and generating a confusion matrix.
  accuracy = accuracy_score(y_test, y_pred_labels)
  precision = precision_score(y_test, y_pred_labels)
  recall = recall_score(y_test, y_pred_labels)

# Grad-CAM Visualization
  Grad-CAM is used to visualize the regions in the images that the model focuses on when making predictions.
  grad_cam_debug(vit_model, X_test[0], 'multi_head_attention_1')

# Results
After running the model, the following results are displayed:
Model Accuracy
Precision and Recall
Confusion Matrix
Grad-CAM visualization for selected test images.

# Files
tumor_data.npz: Contains the training and test data in a compressed format.

# Notes
You can adjust the level of downsampling for .svs images by changing the level parameter.
Ensure the Google Drive paths are correct for your dataset.
For debugging Grad-CAM or other issues, consult the logging outputs from the script.

(generated using ChatGPT)
