# Plant-Disease-Classifier-Web-App

Project Description:

The Plant Disease Classifier is an interactive web application that identifies diseases in plant leaves using a trained Convolutional Neural Network (CNN) model. The project helps farmers, researchers, and plant enthusiasts quickly detect plant diseases from images, enabling timely action to prevent crop damage.

The CNN model is trained on the PlantVillage dataset, which contains thousands of images of healthy and diseased leaves from different plant species. Users can upload an image of a leaf, and the app preprocesses it, predicts the disease class, and displays the result in a user-friendly interface.

Features:
1. Image Upload
- Supports JPG, JPEG, and PNG formats.
- Handles images of varying sizes and automatically resizes them to match the CNN model input.

2. Image Preprocessing

- Resizes images to 224×224 pixels.
- Normalizes pixel values to the [0, 1] range.
- Adds a batch dimension to make images compatible with the CNN model.

3. Disease Prediction
- Uses a trained CNN to predict probabilities for multiple classes of plant diseases.
- Selects the class with the highest probability as the predicted disease.
- Maps numeric class indices to human-readable class names using a JSON file.

4. Interactive Web Interface

- Built with Streamlit for a simple and responsive interface.
- Displays the uploaded image alongside the predicted class.
- Allows users to click a Classify button for instant predictions.

5. Multi-Class Classification Support
- Capable of predicting multiple diseases across various plant species.
- Trained on a diverse dataset to improve accuracy and robustness.

>> Learning Outcomes:

1. Deep Learning & CNNs
2. Designing and training Convolutional Neural Networks for image classification.
3. Understanding convolution, pooling, flattening, and fully connected layers.
4. Handling multi-class classification with softmax activation and categorical cross-entropy loss.

Image Preprocessing
- Resizing, normalizing, and batching images for deep learning models.
- Using Python libraries like Pillow and NumPy to prepare images for training and prediction.

Web App Development
- Building interactive applications with Streamlit.
- Handling user uploads, buttons, and dynamic display of predictions.
- Creating a user-friendly interface for machine learning models.

Model Deployment
- Saving and loading trained TensorFlow models (.h5 format).
- Mapping model outputs to human-readable classes using JSON.
- Integrating machine learning models with real-world applications.

Project Management & Documentation
- Organizing code, models, and resources in a structured way.
- Documenting features, workflow, and results for easier understanding and sharing.

Installation Requirements:

1. Python 3.8 or above
2. TensorFlow
3. Streamlit
4. Pillow
5. NumPy

How to Run the App:

1. Clone the repository and navigate to the folder.
2. Run the Streamlit app using: streamlit run app.py
3. The app will open in the browser. Upload an image of a plant leaf and click Classify to see the prediction.
4. Example:
Uploaded Image: A leaf image
Prediction: Apple___healthy




