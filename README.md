# Remaining Useful Life Prediction - Inception Attention
 ### This is the project of Remaining Useful Life（RUL） prediction based on deep learning technology. 
 ### 基于深度学习的剩余使用寿命（RUL）预测项目。

## 项目
 在本项目中，我们基于Trransformer模型搭建一个名为Inception-Attention的网络模型架构用于实现时间序列相关的剩余使用寿命（RUL）预测问题。在Inception-Attention的模型架构中，我们主要分为三个模块：  
   1）针对不同数据集的数据预处理模块；  
   2）基于多种Masked-Attention机制的Transformer-based的Encoder模块；  
   3）组合式的预测器模块.  
   
 此外，基于机械的平滑退化特性，降低振荡对估计结果造成的影响。我们提出了HTS-Loss Function用于平滑预测结果。他主要由两个模块组成：  
   1）MSE_Smoothness_Loss；  
   2）Weighted_MSE_Loss.  
 
## 数据集介绍
  在本项目中，我们将使用三种不同的数据集：  
    1） NASA Turbofan Jet Engine DataSet -- CMAPSS；  
    2） 2012 IEEE Prognostic Challenge -- FEMTO Bearing Dataset；  
    3） XJTU-SY Bearing Datasets.  
  
  针对于三种不同的数据集，我们首先将针对其进行数据加载与处理，相关代码会分别放在CMAPSS_Related，FEMTO——Related与XJTU_Related对应文件夹内。  

### CMAPSS Dataset
  针对于CMAPSS Dataset。我们首先会根据数据的可视化结果，删除相关性弱的特征。其可视化代码与图片在文件夹Feature_Visualization_CMAPSS中 。此外，根据其特性对Label数据增加一个拐点-MAXLIFE.  
  关于数据集的调用与训练，请在Main.ipynp中使用args.dataset_name = "CMAPSS"，实现数据的加载与训练。
  
### FEMTO Bearing Dataset  
  针对FEMTO Bearing Dataset,我们将从STFT与tsfresh做特征提取中选择一种方法进行数据预处理, 其剩余使用寿命, 水平与纵向加速度的可视化, 以及STFT方法的可视化的图片在文件夹Visualization_FEMTO中。  
  
### XJTU Bearing Dataset  
  该数据集类似于FEMTO Bearing Dataset, 唯一的不同点在于该数据集在相同的采样频率之下,每个时间点采样时间更长,参数更多。 基于FEMTO与XJTU数据类型相似的原因,我们同样采用STFT与tsfresh两种方法对数据进行数据预处理。
