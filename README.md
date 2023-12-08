# Celebrity_Prediction

This project aims to develop a Convolutional Neural Network (CNN) model capable of recognizing celebrities from images. The model utilizes a dataset containing images of five prominent personalities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The dataset is preprocessed, and a CNN architecture is implemented and trained for image classification.

Introduction:
Celebrities' recognition from images holds practical significance in various domains, including entertainment, security, and social media. This project addresses the challenge of accurately identifying celebrities from diverse images.

Data Collection and Preprocessing:
The dataset comprises images of the mentioned celebrities collected from diverse sources. Images undergo preprocessing steps such as resizing to a uniform dimension of 128x128 pixels, normalization, and labeling according to the respective celebrity.

Methodology:
A CNN architecture is employed for this classification task. The model consists of convolutional layers for feature extraction, max-pooling layers for downsampling, a flattening layer, dense layers for classification, and dropout layers for regularization.

Implementation:
The project is implemented in Python using TensorFlow, OpenCV, PIL, NumPy, and other supporting libraries. The codebase is structured into sections for data loading, model construction, training, and evaluation.

Experimentation and Training:
The dataset is split into training and test sets (80/20 ratio). Training involves hyperparameter tuning, including 50 epochs, a batch size of 128, and the Adam optimizer. The model is trained on a GPU-enabled environment.

Evaluation Metrics and Results:
The model achieves an accuracy of 79.41% on the test set. Evaluation metrics including precision, recall, and F1-score are calculated for individual celebrity classes. The report includes a detailed classification report and confusion matrix analysis.

Conclusion:
The developed CNN model demonstrates promising results in classifying celebrities from images. The project underscores the potential and challenges of image-based celebrity recognition, providing a foundation for further refinement and application in real-world scenarios.
