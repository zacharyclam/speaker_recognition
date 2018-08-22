#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : train.py
# @Time     : 2018/8/12 12:49 
# @Software : PyCharm

import os
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Input, MaxoutDense, Merge
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.core import Lambda
import keras.backend as K

import numpy as np

tf.flags.DEFINE_integer("batch_size", default=32, help="Batch size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", default=100, help="Number of training epochs (defaule:100)")
tf.flags.DEFINE_integer("num_classes", default=340, help="Number of training data classes (default:340)")
tf.flags.DEFINE_float("learn_rate", default=0.0001, help="learn rate (default: 0.0001)")
tf.flags.DEFINE_string("category", default="", help="the category of data")
tf.flags.DEFINE_string("model_dir", default="model", help="the model file dir")
tf.flags.DEFINE_string("tensorboard_dir", default="logs", help="the tensorboard file dir")
tf.flags.DEFINE_string("datalist_dir", default="data/bin", help="the data list file dir")

# FLAGS 是一个对象，保存了解析后的命令行参数
FLAGS = tf.flags.FLAGS
# 进行解析
FLAGS.flag_values_dict()

# set parameters
batchSize = FLAGS.batch_size
nClasses = FLAGS.num_classes
nEpochs = FLAGS.num_epochs
learnRate = FLAGS.learn_rate

dataListDir = FLAGS.datalist_dir
category = FLAGS.category
modelDir = FLAGS.model_dir
tensorboardDir = FLAGS.tensorboard_dir

if not os.path.exists(modelDir):
    os.makedirs(modelDir)

if not os.path.exists(tensorboardDir):
    os.makedirs(tensorboardDir)

# the paths
cwd = os.getcwd()  # current working directory
train_path = os.path.join(dataListDir, "train_list.txt")  # train text path
test_path = os.path.join(dataListDir, "test_list.txt")

# count the number of samples
f = open(train_path)
nTrain = len(f.readlines())  # number of train samples
f.close()

f = open(train_path)
nTest = len(f.readlines())  # number of train samples
f.close()


# 生成数据
def generate_fit(path, batch_size, nClasses):
    """

    :param path: 数据路径
    :param batch_size:
    :param nClasses: 类别
    :return:
    """
    with open(path) as f:
        dataList = [line.strip().split(' ') for line in f]
    index = np.arange(len(dataList))
    while True:
        cnt = 0
        X = []
        Y = []
        # shffle data
        np.random.shuffle(index)
        dataList = np.array(dataList)[index, :]
        for bin_path, label in dataList:
            # shape(299,40)
            x = np.fromfile(bin_path, dtype=np.float).reshape((299, 40))

            sepX = [x[i:i + 100].flatten() for i in range(0, 199, 20)]

            X.append(sepX)
            Y.append(label)

            cnt += 1

            if cnt % batch_size == 0:
                yield [np.array(X), np.array(to_categorical(Y, nClasses))]
                X = []
                Y = []
                cnt = 0


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # wget 7545
    # pid 33904
    # nohup python3 -u  train.py --batch_size=64 --num_epochs=1000 --learn_rate=0.0001 --category="train" > logs.out 2>&1 &
    # python train.py --batch_size=32 --num_epochs=200 --num_classes=20 --category="test"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45  # 占用GPU90%的显存
    K.set_session(tf.Session(config=config))
    model = Sequential()

    # (batch_size, 9, 2000)
    model.add(Dense(256, input_shape=(10, 4000), name="dense1"))
    model.add(Activation('relu', name="activation1"))
    # model.add(MaxoutDense((9, 256), nb_feature=2, name="maxout1"))

    model.add(Dense(256, name="dense2"))
    model.add(Activation('relu', name="activation2"))
    # model.add(MaxoutDense(256, nb_feature=2, name="maxout2"))

    model.add(Dense(512, name="dense3"))
    model.add(Activation('relu', name="activation3"))
    # model.add(MaxoutDense(256, nb_feature=2, name="maxout3"))
    model.add(Dropout(0.5, name="drop3"))

    model.add(Dense(1024, name="dense4", kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu', name="activation4"))
    # model.add(MaxoutDense(256, nb_feature=2, name="maxout4"))
    model.add(Dropout(0.5, name="drop4"))

    model.add(Lambda(lambda x: K.mean(x, axis=1), name="mean"))


    modelInput = Input(shape=(10, 4000))
    features = model(modelInput)
    extractFeatureModel = Model(inputs=modelInput, outputs=features)

    categoryPredict = Dense(nClasses, activation='softmax', name="predict")(features)

    srModel = Model(inputs=modelInput, outputs=categoryPredict)



    # srModel = load_model("model/checkpoint-00356-0.97.h5")

    opt = Adam(lr=learnRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    srModel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                  min_lr=1e-6, mode="min", cooldown=5)

    tbCallBack = TensorBoard(log_dir=tensorboardDir,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)

    checkpoint = ModelCheckpoint(filepath=os.path.join(modelDir, "checkpoint-{epoch:05d}-{acc:.2f}.h5"),
                                 monitor='acc', verbose=2, save_best_only=True, mode='max')

    srModel.fit_generator(generate_fit(train_path, batchSize, nClasses),
                          steps_per_epoch=np.ceil(nTrain / batchSize),
                          shuffle=True,
                          validation_data=generate_fit(test_path, batchSize, nClasses),
                          validation_steps=np.ceil(nTest / batchSize),
                          epochs=nEpochs,
                          verbose=1,
                          callbacks=[reduce_lr, checkpoint,tbCallBack]
                          )

    extractFeatureModel.save("spk.h5")
