import numpy as np
import os
from keras.models import load_model
import time

from code.utils.process_wav import get_log_fbank
from code.utils.calculate_cds import get_cds


def split_data(data_dir, usage, enroll_sentence_nums=20, val_sentence_nums=100):
    """

    :paramdataDir: 文件存储路径
    :param usage: 数据集类别
    :param enroll_num: 注册使用语句条数
    :param val_num: 验证使用语句条数
    :return:
    """
    data_path = os.path.join(data_dir, usage)

    # 获取文件列表
    data_list = []
    for dir in os.listdir(data_path):
        file_list = os.listdir(os.path.join(data_path, dir))
        data_list.append([os.path.join(data_path, dir, file) for file in file_list])

    # (list , label)
    enroll_tuple = []
    val_tuple = []

    for i, file_list in enumerate(data_list):
        enroll_tuple.append((file_list[:enroll_sentence_nums], i))
        val_tuple.append((file_list[enroll_sentence_nums:enroll_sentence_nums + val_sentence_nums], i))

    return enroll_tuple, val_tuple


def get_features(model, data_tuple, mean=True):
    def caculate_features(model, fb_input, mean=True):
        features = model.predict(fb_input)
        features = np.array(features)
        if mean:
            # (1,256)
            return np.mean(features, axis=0)
        else:
            # (N,256)
            return features

    fearures = []
    for filelist, label in data_tuple:
        # 读取 wav 文件 并提取fbank特征
        fb_input = [get_log_fbank(bin_path).reshape((299, 40)) for bin_path in filelist]
        # for bin_path in filelist:
        #     fbank = get_log_fbank(bin_path).reshape((11960, 1))
        #     sepX = [fbank[i:i + 100].flatten() for i in range(0, 199, 20)]
        #     fb_input.append(fbank)

        fearures.append(caculate_features(model, np.array(fb_input), mean))
    return fearures


def confusion_matrix_test(data_dir, usage, weight_path, predict_path, enroll_sentence_nums=20, val_sentence_nums=100):
    model = load_model(weight_path)

    enroll_tuple, val_tuple = split_data(data_dir, usage, enroll_sentence_nums=enroll_sentence_nums,
                                         val_sentence_nums=val_sentence_nums)

    start_time = time.time()
    # 得到注册语句的特征向量
    enroll_people = get_features(model, enroll_tuple, mean=True)
    print(time.time() - start_time)

    start_time = time.time()
    # 得到验证语句的特征向量
    val_people = get_features(model, val_tuple, mean=False)
    print(time.time() - start_time)

    # 混淆矩阵
    confusion_matrix = np.zeros((40, 40))

    for idx, val_features in enumerate(val_people):
        for val_feat in val_features:
            predict_list = [get_cds(val_feat, reg) for reg in enroll_people]
            predict_label = np.argmax(predict_list)
            confusion_matrix[idx][predict_label] += 1

    print(confusion_matrix)

    with open(predict_path, "w") as f:
        for i in range(confusion_matrix.shape[0]):
            line = ""
            for j in range(confusion_matrix.shape[1]):
                line += str(confusion_matrix[i][j]).zfill(3) + "  "
            f.write(line + "\n")


if __name__ == "__main__":
    data_dir = "data"
    usage = "dev"
    weight_path = "D:\PythonProject\speakerRecognition\model\spk-00106-0.99.h5"
    predict_path = "predict.txt"
    enroll_sentence_nums = 20
    val_sentence_nums = 100

    confusion_matrix_test(data_dir, usage, weight_path, predict_path, enroll_sentence_nums, val_sentence_nums)
