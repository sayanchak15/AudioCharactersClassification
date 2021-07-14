import sounddevice
from scipy.io.wavfile import write
import librosa
import numpy as np
import noisereduce as nr 
import sys
import tensorflow as tf
from tensorflow import keras
import os, shutil
import time

fs = 22050
seconds = 10
# chars = ['A','B','C','D','E','F','G','H']
chars = ['A','B','C','D','E', 'F', 'G','H','I','J','K','L','M','N','O','P']
reconstructed_model = keras.models.load_model("abcdefghijklmnop_my_model")
pred_labels = []
print("*****************************************")
print("*****************RECORDING***************")
print("*****************************************")
record_voice = sounddevice.rec(int(seconds * fs), samplerate = fs, channels = 2, blocking = False)
sounddevice.wait()
write("out/output.wav", fs, record_voice)

# time.sleep(100)
raw_output = "out/output.wav"
signal, sr = librosa.load(raw_output, sr=22050) # sr * T = 22050 * 2(sec)
voice_nr = nr.reduce_noise(audio_clip=signal,noise_clip=signal, verbose=False)

intervals = librosa.effects.split(voice_nr, top_db=15)
chunks = len(intervals)

equal_intervals = np.zeros((chunks,2), dtype=int)
size_of_chunks = 10000
# equal_intervals[0]
for i in range(chunks):
    chunk_length = intervals[i][1]-intervals[i][0]
#     print("Length", chunk_length)
    pad_size = int((size_of_chunks - chunk_length)/2)
#     print(pad_size)
    equal_intervals[i] = np.array([intervals[i][0]-pad_size,intervals[i][1]+pad_size])

print("here", equal_intervals)
directory = "out/split"

## Remove existing files in the split folder ##

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for i in range(len(equal_intervals)):
        x = equal_intervals[i][0]
        y = equal_intervals[i][1]
        write("out/split/{0}.wav".format(i+1), sr, voice_nr[x: y])
        signal, sr = librosa.load(directory+"/"+str(i+1)+".wav", sr=22050)
        if signal.shape == (10000,):
                n_fft = 2048
                hop_length = 512
                MFCC = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length= hop_length, n_mfcc= 39 )
                X = np.array(MFCC)
                r,w = X.shape
                X_new = np.reshape(X,(1,r,w))
                prob = reconstructed_model.predict(X_new)
                idx = np.argmax(prob)
                print(f'position {i+1} and prediction: {chars[idx]}')
                pred_labels.append(idx)









# files = ['A63','A64','A68','B80','B78', 'B81', 'C72', 'C73', 'C74']


# for filename in os.listdir(directory):
#     if filename.endswith(".wav"): 
#         input = filename
#         signal, sr = librosa.load(directory+"/"+input, sr=22050)
#         # sys.path.append('/path/to/ffmpeg')

#         n_fft = 2048
#         hop_length = 512
#         MFCC = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length= hop_length, n_mfcc= 13 )
#         X = np.array(MFCC)
#         r,w = X.shape
#         # print(X.shape)
#         X_new = np.reshape(X,(1,r,w))
#         # print("NEW SHAPE", X_new.shape)
#         # print("File", input)
#         prob = reconstructed_model.predict(X_new)
#         print(prob)
#         pred_labels.append(np.argmax(prob))

print(pred_labels)
for i in pred_labels:
    print(chars[i])