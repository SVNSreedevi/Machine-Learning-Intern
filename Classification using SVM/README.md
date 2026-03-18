🐱🐶 Cats vs Dogs Classification using SVM
📌 Task Description

This project implements a Support Vector Machine (SVM) to classify images of cats and dogs using the cats_vs_dogs dataset from TensorFlow Datasets.

The dataset is automatically downloaded via tfds.load(), so no manual download is required.

📊 Steps Performed

Load the Cats vs Dogs dataset from TensorFlow Datasets.

Preprocess images: resize to 32×32, normalize pixel values, flatten.

Apply PCA (Principal Component Analysis) to reduce features for faster training.

Train an SVM classifier on the reduced features.

Evaluate the model using accuracy and classification report.

Save and reload the trained SVM + PCA models for testing.

🚀 How to Run the Notebook

Open the provided .ipynb file in Google Colab.

Run all cells from top to bottom.

The dataset will be automatically downloaded the first time it runs.

The notebook will output:

Accuracy score

Classification report

Sample predictions on test images

📂 Files Included

Cats_vs_Dogs_SVM.ipynb → Colab notebook with full implementation.

README.md → This file (instructions + details).

📚 Dataset Source

The dataset is available at:
🔗 Cats vs Dogs – TensorFlow Datasets
