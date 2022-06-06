from flask import Flask, request, jsonify
import numpy as np
import requests
import tensorflow as tf

model = tf.keras.models.load_model("model/DR_Model_latest.h5")

abc = {'Mild': 0, 'Moderate': 1, 'No': 2, 'Proliferative': 3, 'Severe': 4}

app = Flask(__name__)


@app.route('/', methods=['GET'])
def prediction():
    try:
        link = request.args['link']
        print(link)
        file_path = link
        content = requests.get(file_path).content
        content = tf.image.decode_png(content, channels=3)
        content = tf.cast(content, tf.float32)
        content /= 255.0
        content = tf.image.resize(content, [150, 150])
        content = np.expand_dims(content, axis=0)
        content = model.predict(content).round(3)
        content = np.argmax(content)
        for key in abc:
            if content == abc[key]:
                hj = key
                com = {'predict': hj}
                return jsonify(com)

        return 'hello'
    except KeyError:
        return 'Welcome Please Enter like this https://diabeticretinopathydetectionapi.herokuapp.com/?link=Give here the image link'


if __name__ == '__main__':
    app.debug = True
    app.run(port=2000)
