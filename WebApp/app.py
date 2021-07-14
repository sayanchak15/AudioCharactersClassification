from flask import Flask, render_template, request, session, make_response, Response, jsonify
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
import json
import math
from itertools import combinations




app = Flask(__name__)

# chars = ['A','B','C','D','E', 'F', 'G','H','I','J','K','L','M']
# y = [0,1,2,3,4,5,6,7,8,9,10,11,12]
chars = ['A','B','C','D','E', 'F', 'G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
mfcc = 13   
size_of_chunks = 10000
n_fft = 2048
hop_length = 512

reconstructed_model = keras.models.load_model("../atoz_cnn_model")
bow = []
with open('../20k.txt', 'r') as f:
    for word in f:
        for w in word.split():
            bow.append(w)
# reconstructed_model = keras.models.load_model("../abcdefghijklmnop_my_model")


def possible_words(charSet):
   print("CharSet", charSet)
   count = len(charSet)
   my_word=''
   for ab in charSet:
      a,b = ab.split('/')
      my_word = my_word+''.join(a)
   my_word = my_word.lower()
   length = len(my_word)
   print(my_word)
   match_len_bow = [w for w in bow if len(w) == len(my_word)]
   print(len(match_len_bow))

   matched_lists = [[] for i in range(length)]
   comb_list = list(range(length))
   for i in range(length):
      matched_lists[i] = [w for w in match_len_bow if w[i] == my_word[i]]

   comb_list = list(combinations(comb_list , math.ceil(length/2)))
   print(comb_list)
   # list_of_list = []
   elements_in_all = []
   for comb in comb_list:
      list_of_list = []
      for i in list(comb):
         list_of_list.append(matched_lists[i])
      words = list(set.intersection(*map(set, list_of_list)))
      if words:
         elements_in_all.append(words)
      else: continue

   print(elements_in_all)
   final_words = []
   if elements_in_all:
      final_words = list(set.union(*map(set, elements_in_all)))
      print(final_words)
   return final_words

def decode():
   pred_labels = []
   raw_output = "../out/output.wav"
   signal, sr = librosa.load(raw_output, sr=22050) # sr * T = 22050 * 2(sec)
   voice_nr = nr.reduce_noise(audio_clip=signal,noise_clip=signal, verbose=False)

   intervals = librosa.effects.split(voice_nr, top_db=12)
   chunks = len(intervals)

   equal_intervals = np.zeros((chunks,2), dtype=int)
   # equal_intervals[0]
   for i in range(chunks):
      chunk_length = intervals[i][1]-intervals[i][0]
   #     print("Length", chunk_length)
      pad_size = int((size_of_chunks - chunk_length)/2)
   #     print(pad_size)
      equal_intervals[i] = np.array([intervals[i][0]-pad_size,intervals[i][1]+pad_size])

   # print("here", equal_intervals)
   directory = "../out/split"

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
         write("../out/split/{0}.wav".format(i+1), sr, voice_nr[x: y])
         signal, sr = librosa.load(directory+"/"+str(i+1)+".wav", sr=22050)
         print("I", i, signal.shape )
         if signal.shape == (size_of_chunks,):
                  MFCC = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length= hop_length, n_mfcc= mfcc )
                  X = np.array(MFCC)
                  r,w = X.shape
                  X_new = np.reshape(X,(1,r,w,1))
                  prob = reconstructed_model.predict(X_new)
                  # idx = np.argmax(prob)
                  idx1 = np.argsort(prob[0])[-1]
                  idx2 = np.argsort(prob[0])[-2]
                  print(f'position {i+1} and prediction: {chars[idx1]} and prob: {prob}')
                  # pred_labels.append(chars[idx])
                  pred_labels.append((chars[idx1]+'/'+chars[idx2]))
   return pred_labels



@app.route("/")
def hello():
   print ("record")
   return render_template("index.html")


@app.route("/record")   
def record():
   print('Recording')
   fs = 22050
   seconds = 10
   record_voice = sounddevice.rec(int(seconds * fs), samplerate = fs, channels = 1, blocking = False)
   sounddevice.wait()
   write("../out/output.wav", fs, record_voice)
   preds = decode()
   print("Here is some preds", preds)
   words = possible_words(preds)
   return jsonify(result=preds, answer = words) 
   

#background process happening without any refreshing
@app.route('/background_process_test')
def background_process_test():
    print ("Hello")
    return ("nothing")


if __name__ == '__main__':
   # chars = ['A','B','C','D','E', 'F', 'G','H','I','J','K','L','M','N','O','P']
   # reconstructed_model = keras.models.load_model("abcdefghijklmnop_my_model")
   app.run()