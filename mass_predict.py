import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import librosa
import os


os. getcwd()
reconstructed_model = keras.models.load_model("abcdefghijklmnop_my_model")

# chars = ['A','B','C','D']
chars = ['A','B','C','D','E','F','G','H']

def get_labels():
    test_labels = []
    directory = "audio/baba/test/"
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            letter = filename[0]
            test_labels.append(chars.index(letter))
    return test_labels



# files = ['A63','A64','A68','B80','B78', 'B81', 'C72', 'C73', 'C74']
pred_labels = []
directory = "audio/baba/test/"
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        input = filename
        signal, sr = librosa.load(directory+"/"+input, sr=22050)
        
        # sys.path.append('/path/to/ffmpeg')
        if signal.shape == (10000,):
                n_fft = 2048
                hop_length = 512
                MFCC = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length= hop_length, n_mfcc= 39 )
                X = np.array(MFCC)
                r,w = X.shape
                # print(X.shape)
                X_new = np.reshape(X,(1,r,w))
                # print("NEW SHAPE", X_new.shape)
                # print("File", input)
                prob = reconstructed_model.predict(X_new)
                pred_label = np.argmax(prob)
                print("input", input)
                print("prediction", chars[pred_label], prob)
                pred_labels.append(pred_label)
        else: continue

# labels = get_labels()
# print(labels)
print(pred_labels)
# x = np.array(labels)
# y = np.array(pred_labels)
# print("Accuracy", np.average(x == y))

# print("Individual Accuracy")
# for i in chars:
#     get_predicted_labels