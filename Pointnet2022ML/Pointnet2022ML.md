## Pointnet2022ML

### 文件概述

- data：包括原始数据和h5预处理格式数据，训练和测试全部转换为h5格式导入
- log：训练结果
  - checkpoints：训练得到的最佳模型（epoch=200，batch size=32）
  - result_fig：包括accuracy、confusion matrix和miou随着训练的变化趋势图
  - reconstruct_fig：训练后的模型和原始数据图
    - 后缀为final：训练模型的预测分类结果
    - 后缀为groundtruth：原始数据图
- testpath：参与模型预测输出预测分类结果对应的原始数据（包括16类物品各随机选取）

### 运行方法

#### 训练

- 打开pointnet_train.py，运行后输入epoch和batch size大小，分别回车开始训练，注意运行前根据实际情况指定GPU：

```python
import os
# select CUDA core before run the train code if there are multiple GPU accessible
os.environ['CUDA_VISIBLE_DEVICES'] = '#'  # is your gpu number
```

- 训练完成后可能存在以下报错，这是因为图片过大导致pycharm的sciview无法显示图片，无视并耐心等待图片写入路径即可

```python
Error: failed to send plot to http://127.0.0.1:63342
```

- 训练的过程日志位于log文件夹下的word文档：train_log.docx

#### 验证

- 运行groundtruth.py，即可得到本次参与点云重构的原始数据图
- 运行txtTOh5file.py，将原始数据进行采样后得到h5文件
- 运行reconfig_data.py，即可将模型预测的可视化结果写入文件中，该步骤也可能出现以下报错，无视并耐心等待图片写入文件即可

```python
Error: failed to send plot to http://127.0.0.1:63342
```

