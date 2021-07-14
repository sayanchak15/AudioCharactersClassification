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

char = 'Z'
copies = 1
src = "audio/baba/" + char + "/source/"
dst = "audio/baba/" + char + "/wav/"

for filename in os.listdir(src):
    if filename.endswith(".mp3"): 
        for i in range(copies):
            sound = AudioSegment.from_mp3(src+filename)
            num = re.search('[A-Z](.*).mp3', filename).group(1)
            print(num)
            dstfile = dst + char + str(num) + '-' + str(i) +'.wav'
            sound.export(dstfile, format="wav",parameters=["-c:a", "pcm_u8"])


        