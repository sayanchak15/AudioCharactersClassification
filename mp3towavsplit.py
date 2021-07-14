import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr 
import sys
from pydub import AudioSegment
import speech_recognition as sr
from numpy import inf
from pydub import AudioSegment
from pydub.playback import play
from pydub.silence import split_on_silence
import scipy.io.wavfile
import re
import os

char = 'H'
src = "audio/baba/" + char + "/"
dst = "audio/baba/" + char + "/"
out_folder = "audio/baba/" + char + "/small/"

for filename in os.listdir(src):
    if filename.endswith(".mp3"): 
        sound = AudioSegment.from_mp3(src+filename)
        num = re.search('[A-Z](.*).mp3', filename).group(1)
        print(num)
        dstfile = dst + char + str(num)+'.wav'
        sound.export(dstfile, format="wav",parameters=["-c:a", "pcm_u8"])

        signal3, sr3 = librosa.load(dstfile, sr=22050) # sr * T = 22050 * 2(sec)
        voice_nr = nr.reduce_noise(audio_clip=signal3,noise_clip=signal3, verbose=False)

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

        for i in range(len(intervals)):
            x = equal_intervals[i][0]
            y = equal_intervals[i][1]
            scipy.io.wavfile.write("{0}/{1}-{2}-{3}.wav".format(out_folder, char ,num, i+1), sr3, voice_nr[x: y])