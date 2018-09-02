# Speaker Recognition 

[![avatar](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/houzhengzhang/speaker_recognition/pulls)
[![avatar](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges)
[![avatar](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![avatar](https://img.shields.io/badge/license-Apache_2-blue.svg)](https://github.com/houzhengzhang/speaker_recognition/blob/master/LICENSE)

​	使用数据集：AISHELL-ASR0009-OS1  [下载](https://pan.baidu.com/s/1dFKRLwl#list/path=%2F)

​	  模型结构参考论文 “DEEP NEURAL NETWORKS FOR SMALL FOOTPRINT TEXT-DEPENDENT“，在实现时将论文中所提出的4层DNN结构的前两层替换为两层一维卷积，训练时通过Softmax分类器进行训练，注册及验证时将Softmax层去掉，DNN的输出作为d-vector,通过计算 *cosine-distance*  来判别说话人是否在注册集内。

* #### 项目结构

  ```
  - code
    -- 0-input				          # 数据预处理
    -- 1-development			          # 模型定义及训练
    -- 2-enrollment			          # 注册
    -- 3-evalution			          # 陌生人验证评估
    -- 4-roc_curve			          # 绘制ROC曲线图，并计算EER及阈值
    -- utils				    
  - data					           # 数据存放
  - docs					           # 参考论文
  - logs					           # tensorboard 日志文件
  - model						   # 模型存储文件
  - results					
    -- features				           # 根据模型计算出注册人及陌生人的d-vector
    -- plots					   # 绘制完成的ROC曲线图
    -- scores					   # 绘制ROC曲线所需的score
  ```

  

* #### 训练

  * 首先对下载好的数据集进行VAD处理，处理代码位于 code/0-input/vad.py 

    *usage:*

    ```shell
    python vad.py --save_dir="../../data/vad_data"  --data_dir="解压之后的数据集路径" \
    --category="要处理的数据类别，eg：train,test,dev" 
    ```

  * 将vad处理后的数据提取 *log fbank* 特征，该过程使用 python_speech_features 库完成

    usage:

    ```shell
    python process_data.py --data_dir="../../data/vad_data"  --save_dir="提取log fbank后 bin文件保存路径" \
    --category="要处理的数据类别" \ 
    --validata_scale="若处理训练集数据，该参数可设置为验证集所占比例，eg：0.05， 若处理其他类别数据将其设置为0即可"
    ```

  * 将训练集和验证集数据文件路径写入txt中，方便训练时打乱数据送入模型

    usage:

    ```shell
    python get_data_list.py --save_dir="../../data/bin/" --category="validate" # 验证集list
    python get_data_list.py --save_dir="../../data/bin/" --category="train"    # 训练集list
    ```

  * 通过执行train.py即可开始训练

    usage:

    ```shell
    python train.py --batch_size=128 --num_epochs=1000 --learn_rate=0.0001
    ```

* #### 评估模型

     直接运行model_test.sh 脚本即可绘制ROC曲线图并计算EER，该脚本需要模型的路径参数，结果文件会保存至results目录下

   usage:

   ```
   model_test.sh "model/checkpoint-00484-0.99.h5"
   ```

* #### 模型参数统计

  Total params: 5,781,524

  Trainable params: 5,781,428

  Non-trainable params: 96

  模型结构图可在 results 目录下查看

* #### 实验结果：

   使用340人的语音数据进行训练，训练完成后，使用dev数据集共40人进行注册，将数据分为注册和验证两部分，每人选取15s音频进行注册，然后用100条长度为1s的音频进行验证，统计TP和FP的个数。使用test数据集共20人进行陌生人验证，每人选取100条长度为1s的音频，统计每条音频与注册集内得分最高的cds值。通过上述测试数据绘制ROC曲线图，计算出EER为12.2%，阈值为0.7824.

<img src="https://github.com/houzhengzhang/speaker_recognition/blob/master/results/plots/checkpoint-00484-0.99.jpg" width="500" hegiht="313" align=center />

- #### 模型下载：

  百度网盘地址：https://pan.baidu.com/s/1wdBvyBkRwC0TyX9R2xnmLw		密码：2w3q

  *其中 checkpoint-00484-0.99.h5 文件包含Softmax层 ，checkpoint-00484-0.99_notop.h5 已去掉Softmax层* 

  模型训练时的超参数：batch_size=128 ，learn_rate=0.0001，实验时使用一块 Titan 进行训练，大约5小时训练完成

