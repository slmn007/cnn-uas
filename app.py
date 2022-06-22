from flask import Flask, render_template, request

import numpy as np
from PIL import Image
import tensorflow

app = Flask(__name__)

model_path = "model/caltech_101.hdf5"
model_train = tensorflow.keras.models.load_model(model_path)

class_dict = {  'Faces': 0,
                'Faces_easy': 1,
                'cougar_body': 2,
                'cougar_face': 3,
                'crocodile': 4,
                'crocodile_head': 5,
                'emu': 6,
                'flamingo': 7,
                'ibis': 8,
                'pigeon': 9,
                'rooster': 10   }

def model_predict(img_path, model_train):
    test_image = Image.load_img(img_path, target_size=(140, 140)) # load data
    test_image = Image.img_to_array(test_image)
    test_image = np.array([test_image])
    result = model_train.predict_on_batch(test_image) # Predict data
    
    return result

@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predic():
    img_file = request.files['img_file']
    img_path = "./images/" + img_file.filename
    img_file.save(img_path)

    result = model_predict(img_path, model_train)

    for category, index in class_dict:
        if index == result.argmax():
            print('Kelas',result.argmax(), 'Menunjukkan Kategori', category)
            
            return render_template('index.html', prediction=category)

if __name__ == '__main__':
    
    app.run()