import os
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Set TensorFlow environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, static_url_path='/static', static_folder='D:/WT_labs/prj/static')

# Load the trained model
model_path = 'D:/WT_labs/prj/model/BHC.h5'
model = load_model(model_path)
print("Model loaded successfully")

# Define the image dimensions expected by the model
image_height, image_width = 224, 224

# Define batch size for data generator
batch_size = 32

# Define data augmentation parameters for training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

# Define ImageDataGenerator for training data
train_generator = train_datagen.flow_from_directory(
    'D:/SEM6/miniproject/Training',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define ImageDataGenerator for testing data
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    'D:/SEM6/miniproject/Test',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get uploaded image file
        img_file = request.files['image']
        if img_file:
            # Save image to a temporary location
            img_path = 'temp.jpg'
            img_file.save(img_path)

            # Load and preprocess the image
            img = load_img(img_path, target_size=(image_height, image_width))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make prediction for uploaded image
            uploaded_image_prediction = model.predict(img_array)[0]
            uploaded_image_predicted_class_index = np.argmax(uploaded_image_prediction)
            labels = train_generator.class_indices
            uploaded_image_predicted_class_label = next(k for k, v in labels.items() if v == uploaded_image_predicted_class_index)

            # Make prediction for test data
            test_predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
            test_predicted_class_indices = np.argmax(test_predictions, axis=1)
            test_labels = test_generator.class_indices
            test_predicted_class_labels = [k for k, v in test_labels.items() if v in test_predicted_class_indices]

            return render_template('result.html',
                                   uploaded_image_prediction=uploaded_image_predicted_class_label,
                                   test_predictions=test_predicted_class_labels)

if __name__ == '__main__':
    app.run(debug=True)

