#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : train.py
# @Time     : 2018/8/12 12:49 
# @Software : PyCharm

import os
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras.backend as K
import numpy as np

from model import construct_model
from data_feeder import generate_fit


root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

tf.flags.DEFINE_integer(
    "batch_size", default=128,
    help="Batch size (default: 128)")

tf.flags.DEFINE_integer(
    "num_epochs", default=100,
    help="Number of training epochs (defaule:100)")

tf.flags.DEFINE_integer(
    "num_classes", default=340,
    help="Number of training data classes (default:340)")

tf.flags.DEFINE_float(
    "learn_rate", default=0.0001,
    help="learn rate (default: 0.0001)")

tf.flags.DEFINE_string(
    "category", default="train",
    help="the category of data")

tf.flags.DEFINE_string(
    "model_dir", default=os.path.join(root_dir, "model"),
    help="the model file dir")

tf.flags.DEFINE_string(
    "tensorboard_dir", default=os.path.join(root_dir, "logs"),
    help="the tensorboard file dir")

tf.flags.DEFINE_string(
    "datalist_dir", default=os.path.join(root_dir, "data/bin"),
    help="the data list file dir")

# FLAGS 是一个对象，保存了解析后的命令行参数
FLAGS = tf.flags.FLAGS
# 进行解析
FLAGS.flag_values_dict()

if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

if not os.path.exists(FLAGS.tensorboard_dir):
    os.makedirs(FLAGS.tensorboard_dir)

# the paths
train_path = os.path.join(FLAGS.datalist_dir, "train_list.txt")
test_path = os.path.join(FLAGS.datalist_dir, "validate_list.txt")

# count the number of samples
f = open(train_path)
train_nums = len(f.readlines())  # number of train samples
f.close()

f = open(test_path)
test_nums = len(f.readlines())  # number of train samples
f.close()

if __name__ == '__main__':
    # 指定使用显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.90  # 占用GPU90%的显存
    K.set_session(tf.Session(config=config))

    # 创建模型
    extract_feature_model, sr_model = construct_model(FLAGS.num_classes)

    # 创建优化器
    opt = Adam(lr=FLAGS.learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sr_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # 学习率衰减
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                  min_lr=1e-8, mode="min", cooldown=10, verbose=1)

    tbCallBack = TensorBoard(log_dir=FLAGS.tensorboard_dir,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)

    checkpoint = ModelCheckpoint(filepath=os.path.join(FLAGS.model_dir, "checkpoint-{epoch:05d}-{val_acc:.2f}.h5"),
                                 monitor='val_acc', verbose=2, save_best_only=True, mode='max')

    # 开始训练
    sr_model.fit_generator(generate_fit(train_path, FLAGS.batch_size, FLAGS.num_classes),
                           steps_per_epoch=np.ceil(train_nums / FLAGS.batch_size),
                           shuffle=True,
                           validation_data=generate_fit(test_path, FLAGS.batch_size, FLAGS.num_classes),
                           validation_steps=np.ceil(test_nums / FLAGS.batch_size),
                           epochs=FLAGS.num_epochs,
                           verbose=2,
                           callbacks=[reduce_lr, checkpoint, tbCallBack]
                           )

    sr_model.save("spk.h5")

    # usage
    # nohup python3 -u  train.py --batch_size=128 --num_epochs=1000 --learn_rate=0.0001  > logs.out 2>&1 &
