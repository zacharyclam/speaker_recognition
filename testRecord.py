#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : testRecord.py
# @Time     : 2018/8/20 21:10 
# @Software : PyCharm
from keras.models import load_model
import numpy as np
from code.utils.process_wav import get_log_fbank
from code.utils.calculate_cds import get_cds

fbank1 = get_log_fbank("record/01.wav")
fbank2 = get_log_fbank("record/02.wav")
fbank3 = get_log_fbank("record/change02.wav")

model = load_model("D:\PythonProject\speakerRecognition\model\spk.h5")

features1 = model.predict(fbank1.reshape((11960, 1))[np.newaxis, :, :])
features2 = model.predict(fbank2.reshape((11960, 1))[np.newaxis, :, :])
features3 = model.predict(fbank3.reshape((11960, 1))[np.newaxis, :, :])

print(get_cds(features1, features2))
print(get_cds(features1, features3))