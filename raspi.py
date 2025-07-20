import numpy as np
import librosa
import pickle
import tensorflow as tf
from IPython.display import Audio,Image

with open("map_name.plk", "rb") as f:
    lb=pickle.load(f)

#load tf
interpreter = tf.lite.Interpreter("cat_dog.tflite")
interpreter.allocate_tensors()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_audio(fn):
    # Load and process audio
    audio, s_rate = librosa.load(fn)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=s_rate, n_mels=128).T, axis=0)
    mel = mel.reshape(1, 128, 1, 1).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], mel)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_index = np.argmax(output_data)
    class_name = lb.classes_[pred_index]
    
    print(f"Predicted pet: {class_name}")

    # Show image based on prediction
    if class_name.lower() == 'dogs':
        Image("/kaggle/input/daonjdjd/dog.jpeg", width=200)
    elif class_name.lower() == 'cats':
        Image('/kaggle/input/daonjdjd/cat.jpeg', width=200)


    return Audio(data= audio, rate= s_rate)

see= predict_audio('sample/dog_barking_1.wav')
see