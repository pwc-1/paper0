"""
######################## Attention!  ########################
使用注意事项：
1、本示例需要用户定义的参数有--multi_data_url,这个参数任务中是必选的，model_url参数是可选的
具体的含义如下：
--multi_data_url是启智平台上选择的数据集的obs路径
--model_url是训练结果回传到启智平台的obs路径
2、用户需要调用OpenI.C2NETMultiDatasetToEnv等函数，来实现数据集、预训练模型文件的拷贝
3、智算网络区别于启智：
(1)智算的数据集拷贝到训练镜像后需要解压，请使用C2NETMultiDatasetToEnv函数
(2)智算任务结果不需要用户调用函数回传，需要将结果输出到/cache/output文件夹下，才能在启智平台下载结果
在某些特殊情况下，若用户想要手动上传结果，可以使用model_url参数回传结果到启智平台，model_url用法与启智集群的train_url参数用法一样
"""

import os
import argparse

from config import mslt_cfg as cfg
# from dataset_distributed import create_dataset_parallel
from dataset import create_dataset
from model_structure.MSLT import MSLT
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
import time
from openi import c2net_multidataset_to_env as DatasetToEnv    
from openi import env_to_openi        
from openi import EnvToOpenIEpochEnd   
import mindspore as ms
import numpy as np 
from mindspore import Tensor

parser = argparse.ArgumentParser(description='MindSpore Lenet Example')

parser.add_argument('--multi_data_url',
                    help='必选；使用数据集，需要定义的参数',
                    default= '[{}]')            
parser.add_argument('--model_url',
                    help='可选；需要手动回传结果到启智才需要定义的参数',
                    default= '')                                 

parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'CPU'],
    help='device where the code will be implemented (default: Ascend),if to use the CPU on the Qizhi platform:device_target=CPU')

parser.add_argument('--epoch_size',
                    type=int,
                    default=200,
                    help='Training epochs.')

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    data_dir = '/cache/data'  
    train_dir = '/cache/output'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    ###Initialize and copy data to training image
    #DatasetToEnv(args.multi_data_url, data_dir)
    device_num = 1
    #使用单卡时
    if device_num == 1:
        DatasetToEnv(args.multi_data_url,data_dir)
        context.set_context(mode=context.GRAPH_MODE,device_target=args.device_target)
        # context.set_context(mode=context.PYNATIVE_MODE,device_target=args.device_target)
        #使用数据集的方式  
        ds_train = create_dataset(image_path="/dataset/train/train_images", label_path="/dataset/train/label_images", batch_size=cfg.batch_size)

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
    # breakpoint()
    network = MSLT()
    # param_dict = ms.load_checkpoint("./checkpoint_mslt-iter17000.ckpt")
    # ms.load_param_into_net(network, param_dict)
    net_loss = nn.MSELoss(reduction='mean')
    net_opt = nn.Adam(network.trainable_params(), learning_rate=cfg.lr, beta1=0.9, beta2=0.999)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

    if args.device_target != "Ascend":
        model = Model(network,net_loss,net_opt)
    else:
        model = Model(network, net_loss,net_opt,amp_level="O2")
    config_ck = CheckpointConfig(save_checkpoint_steps=int(cfg.save_checkpoint_steps / device_num),
                                keep_checkpoint_max=cfg.keep_checkpoint_max)
    #Note that this method saves the model file on each card. You need to specify the save path on each card.
    # In this example, get_rank() is added to distinguish different paths.
    if device_num == 1:
        outputDirectory = train_dir 
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
    # set callback functions
    callback =[time_cb,LossMonitor()]
    local_rank=0#int(os.getenv('RANK_ID'))
    #非必选，每个epoch结束后，都手动上传训练结果到启智平台，注意这样使用会占用很多内存，只有在部分特殊需要手动上传的任务才需要使用
    uploadOutput = EnvToOpenIEpochEnd(train_dir,args.model_url)
    callback.append(uploadOutput) 
    # for data parallel, only save checkpoint on rank 0
    if local_rank==0 :
        callback.append(ckpoint_cb) 
    
    model.train(epoch_size,ds_train,callbacks=callback)