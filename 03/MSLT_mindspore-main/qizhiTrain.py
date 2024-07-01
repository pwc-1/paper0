"""
示例选用的数据集是MNISTData.zip
数据集结构是：
 MNISTData.zip
  ├── test
  │   ├── t10k-images-idx3-ubyte
  │   └── t10k-labels-idx1-ubyte
  └── train
      ├── train-images-idx3-ubyte
      └── train-labels-idx1-ubyte 
      
使用注意事项：
1、本示例需要用户定义的参数有--multi_data_url,--train_url,--device_target，这3个参数任务中必须定义
具体的含义如下：
--multi_data_url是启智平台上选择的数据集的obs路径
--train_url是训练结果回传到启智平台的obs路径
2、用户需要调用openi.py下的openi_multidataset_To_env,pretrain_to_env,env_to_openi等方法，来实现数据集、预训练模型文件、训练结果的拷贝和回传
"""

import os
import argparse
from config import mnist_cfg as cfg
from dataset0 import create_dataset
from model_structure.MSLT import MSLT
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
import time
from openi import openi_multidataset_to_env as DatasetToEnv
from openi import env_to_openi
from openi import EnvToOpenIEpochEnd


parser = argparse.ArgumentParser(description='MindSpore Lenet Example')


parser.add_argument('--multi_data_url',
                    help='使用数据集训练时，需要定义的参数',
                    default= '[{}]')                        

parser.add_argument('--train_url',
                    help='回传结果到启智，需要定义的参数',
                    default= '')                          

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: Ascend),if to use the CPU on the Qizhi platform:device_target=CPU')

parser.add_argument('--epoch_size',
                    type=int,
                    default=5,
                    help='Training epochs.')

if __name__ == "__main__":
    ###请在代码中加入args, unknown = parser.parse_known_args()，可忽略掉--ckpt_url参数报错等参数问题
    args, unknown = parser.parse_known_args()
    data_dir = '/cache/data'  
    train_dir = '/cache/output'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # ###拷贝数据集到训练环境
    # DatasetToEnv(args.multi_data_url, data_dir)
  
    device_num = 1
    #使用单卡时
    if device_num == 1:
        ###拷贝数据集到训练环境
        context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)
        DatasetToEnv(args.multi_data_url,data_dir)
        #使用数据集的方式  
        ds_train = create_dataset(image_path=os.path.join(data_dir + "/train/train_images"), label_path=os.path.join(data_dir + "/train/label_images"), batch_size=cfg.batch_size)
    #使用多卡时        
    # if device_num > 1:
    #     # set device_id and init for multi-card training
    #     context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
    #     context.reset_auto_parallel_context()
    #     context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
    #     init()
    #     #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
    #     local_rank=int(os.getenv('RANK_ID'))
    #     if local_rank%8==0:
    #         ###拷贝数据集到训练环境
    #         DatasetToEnv(args.multi_data_url,data_dir)
    #         #Set a cache file to determine whether the data has been copied to obs. 
    #         #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    #         f = open("/cache/download_input.txt", 'w')    
    #         f.close()
    #         try:
    #             if os.path.exists("/cache/download_input.txt"):
    #                 print("download_input succeed")
    #         except Exception as e:
    #             print("download_input failed")
    #     while not os.path.exists("/cache/download_input.txt"):
    #         time.sleep(1)               
    #     ds_train = create_dataset_parallel(os.path.join(data_dir + "/MNISTData", "train"),  cfg.batch_size)

    network = MSLT()
    net_loss = nn.MSELoss(reduction='mean')
    net_opt = nn.Adam(network.trainable_params(), learning_rate=cfg.lr, beta1=0.9, beta2=0.999)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    if args.device_target != "Ascend":
        model = Model(network,
                      net_loss,
                      net_opt)
    else:
        model = Model(network,
                      net_loss,
                      net_opt,
                      amp_level="O2")

    config_ck = CheckpointConfig(
        save_checkpoint_steps=cfg.save_checkpoint_steps,
        keep_checkpoint_max=cfg.keep_checkpoint_max)
    #Note that this method saves the model file on each card. You need to specify the save path on each card.
    # In this example, get_rank() is added to distinguish different paths.
    if device_num == 1:
        outputDirectory = train_dir + "/"
    if device_num > 1:
        outputDirectory = train_dir + "/" + str(get_rank()) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_mslt",
                                directory=outputDirectory,
                                config=config_ck)
    print("============== Starting Training ==============")
    epoch_size = cfg['epoch_size']
    if (args.epoch_size):
        epoch_size = args.epoch_size
        print('epoch_size is: ', epoch_size)
    #Custom callback, upload output after each epoch
    uploadOutput = EnvToOpenIEpochEnd(train_dir,args.train_url)
    model.train(epoch_size, ds_train,callbacks=[time_cb, ckpoint_cb,LossMonitor(), uploadOutput])

    ###上传训练结果到启智平台   
    env_to_openi(train_dir,args.train_url)
