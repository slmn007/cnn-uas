from flask import Flask, render_template, request

import numpy as np

import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

model_path = "model/caltech_101.hdf5"
model_train = load_model(model_path)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('.\Test',
                                            target_size=(140, 140),
                                            batch_size= 32,
                                            class_mode='categorical')

def model_predict(img_path, model_train):
    test_image = tensorflow.keras.utils.load_img(img_path, target_size=(140, 140)) # load data
    test_image = tensorflow.keras.utils.img_to_array(test_image)
    test_image = np.array([test_image])
    result = model_train.predict_on_batch(test_image) # Predict data
    
    return result

@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predic():
    img_file = request.files['img_file']
    img_path = "images/" + img_file.filename
    print(img_path)
    print('woy sat disini')
    img_file.save(img_path)

    result = model_predict(img_path, model_train)

    for category, index in test_set.class_indices.items():
        if index == result.argmax():
            print('Kelas',result.argmax(), 'Menunjukkan Kategori', category)
            
            return render_template('index.html', prediction=category)

if __name__ == '__main__':
    
    app.run()