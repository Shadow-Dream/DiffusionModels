环境配置：
（1）conda：本目录下requirements.yml文件记录了解释模型代码所需的python库环境
安装有pytorch 1.13+，opencv-python库的环境，应能按照报错提示安装缺漏的库以运行代码
（2）docker：若使用docker环境，请使用docker-compose安装nvcr.io/nvidia/pytorch:23.02-py3镜像，并手动安装opencv-python

模型简介：
（1）Denoising Diffusion Probabilistic Model 复现了最基础的DDPM训练代码与简单测试程序
（2）Improved Denoising Diffusion Probabilistic Model 复现了IDDPM的Linear Schedule、Cosine Schedule、DDIM与Respacing等方法
（3）Consistency Model 复现了基于欧拉微分与MSE损失的Consistency Model中的Consistency Training与少步生成
（4）Consistency Cold Diffusion Model 复现了基础的Cold Diffusion模型训练方法与Consistency  Cold Diffusion训练方法

运行指南：
（1）出于压缩文件大小考虑，数据集文件没有附带在源码文件中：
下载地址：https://mo.zju.edu.cn/explore/6076b3efd696542fca5d0e59?type=dataset
运行模型前，请在对应模型的utils/loader.py脚本中修改数据集路径
（2）由于代码遗留问题与模型特性原因，仅有Consistency Cold Diffusion Model源码中应用了数据集异常图片过滤和预处理
（3）模型源码中，train.py为模型训练脚本。Consistency Cold Diffusion Model中还附带consistency_train.py脚本，用以使用Consistency Model的方法训练
出于调试方便，源码中并未附带自动调节学习率等结构，需要根据情况手动调节
（4）Consistency Cold Diffusion中，Consistency函数的c0和c1参数的构造为get_coef函数根据time steps产生，为人为构造
源码中附带了一个我认为较为合理的构造方式。构造需满足当t趋近于0，c0趋近于1；t趋近于T，c0趋近于0
（5）模型源码中，test.py是为模型的可视化调试预留的脚本，仅包含部分生成方式，需要手动调节来控制生成过程，如：
在IDDPM源码中，需要手动选择diffusion.py中的生成方式，并调节Respacing步骤
在CCD源码中，需要将Consistency Function替换model进行生成
（6）训练的各项参数均可调节，部分参数并未以全局变量形式标注，如UNet中间层的通道数等，需要在构造函数中手动进行修改