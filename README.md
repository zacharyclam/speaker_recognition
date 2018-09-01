# Speaker Recognition

![avatar](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![avatar](https://badges.frapsoft.com/os/v2/open-source.png?v=103)
![avatar](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![avatar](https://img.shields.io/badge/license-Apache_2-blue.svg)

​	使用数据集：AISHELL-ASR0009-OS1  

​	模型结构参考论文 “DEEP NEURAL NETWORKS FOR SMALL FOOTPRINT TEXT-DEPENDENT“，在实现时将论文中所提出的4层DNN结构的前两层替换为两层一维卷积，训练时通过Softmax分类器进行训练，注册及验证时将Softmax层去掉，DNN的输出作为d-vector,通过计算 *cosine-distance*  来判别说话人是否在注册集内。

* 模型参数统计

  Total params: 5,781,524
  Trainable params: 5,781,428
  Non-trainable params: 96

  模型结构图在speakerRecognition/results/ 目录下

* 模型下载：

  百度网盘地址：https://pan.baidu.com/s/1wdBvyBkRwC0TyX9R2xnmLw		密码：2w3q

  *其中 checkpoint-00484-0.99.h5 文件包含Softmax层 ，checkpoint-00484-0.99_notop.h5 已去掉Softmax层* 

* 实验结果：

​	使用340人的语音数据进行训练，训练完成后，使用dev数据集共40人进行注册，将数据分为注册和验证两部分，每人选取15s音频进行注册，然后用100条长度为1s的音频进行验证，统计TP和FP的个数。使用test数据集共20人进行陌生人验证，每人选取100条长度为1s的音频，统计每条音频与注册集内得分最高的cds值。通过上述测试数据绘制ROC曲线图，计算出EER为12.2%，阈值为0.7824.

<img src="https://github.com/houzhengzhang/speaker_recognition/blob/master/results/plots/checkpoint-00484-0.99.jpg" width="500" hegiht="313" align=center />
