# Breast-Cancer-Detection
 ## Breast Cancer Detection and Classification

Breast cancer is a critical health concern, and early diagnosis is essential for effective treatment. This project focuses on developing a deep learning model for breast cancer detection and classification using ultrasound images. The model is designed to classify images into three categories: benign, malignant, and normal, aiding in early diagnosis and intervention.

### Dataset
The dataset consists of ultrasound images of breast tissues, with each image labeled as benign, malignant, or normal. The dataset has been preprocessed and augmented to ensure diversity and improve the model's performance.You can download the Dataset from [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)

### Model Development
The breast cancer detection model is built using a convolutional neural network (CNN) based on the Keras framework. The ResNet50 architecture is used as the base model, with additional custom layers for classification. The model has been trained using Adam optimizer and balanced class weights to mitigate the data imbalance. The training process involves utilizing image augmentation techniques to enhance the diversity of the dataset.

### Model Evaluation
The model’s performance is evaluated using various metrics, including accuracy, classification report, and confusion matrix, to assess its ability to classify breast cancer ultrasound images accurately.

### Streamlit Integration
In addition to the model development and training, a Streamlit web app has been implemented to provide an interactive platform for breast cancer prediction. The application allows users to upload ultrasound images, and the model predicts the likelihood of benign, malignant, or normal tissue, providing valuable insights and supporting medical professionals in their decision-making process.

### How to Use the Streamlit Web App
1. **Open the Web App**: Upon accessing the Streamlit web app, users will be prompted with an introductory interface providing a brief overview of the application’s purpose and functionality.

2. **Image Upload**: Users are then invited to upload ultrasound images they would like to analyze for breast cancer classification.

3. **Prediction**: Once an image is uploaded, the model processes the image and predicts the likelihood of the tissue being benign, malignant, or normal. The results are displayed to the user, providing valuable insights and potential indicators for further clinical assessment.

### Conclusion
This project showcases the application of deep learning in medical imaging for breast cancer detection. By integrating the trained model into a user-friendly web app, the project aims to provide accessible and convenient tools for healthcare professionals in their efforts to diagnose and treat breast cancer effectively.

The code, dataset, and web app provide valuable resources for further research and development in the field of medical imaging and deep learning applications in healthcare.
