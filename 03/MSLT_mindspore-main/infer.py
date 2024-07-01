
"""
示例选用的模型文件是：checkpoint_lenet-1_1875.ckpt

使用注意事项：
1、本示例需要用户定义的参数有--multi_data_url,--pretrain_url,--result_url，这3个参数任务中必须定义
具体的含义如下：
--multi_data_url是启智平台上选择的数据集的obs路径
--pretrain_url是启智平台上选择的预训练模型文件的obs路径
--result_url是训练结果回传到启智平台的obs路径
2、用户需要调用OpenI.py下的DatasetToEnv,PretrainToEnv,UploadToOpenI等函数，来实现数据集、预训练模型文件、训练结果的拷贝和回传
"""

import os
import argparse
import mindspore.nn as nn
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore import Tensor
from dataset0 import create_dataset
from config import mslt_cfg as cfg
from model_structure.MSLT import MSLT
from PIL import Image
from openi import openi_multidataset_to_env as DatasetToEnv
from openi import env_to_openi
from openi import pretrain_to_env


parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
parser.add_argument('--multi_data_url',
                type=str,
                default= '[{}]',
                help='path where the dataset is saved')      
parser.add_argument('--pretrain_url',
                help='model to save/load',
                default=  '[{}]')  
parser.add_argument('--result_url',
                help='result folder to save/load',
                default= '')   
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')                

if __name__ == "__main__":            
    args, unknown = parser.parse_known_args()

    ###Initialize the data and result directories in the inference image###
    data_dir = '/cache/data'  
    pretrain_dir = '/cache/pretrain'
    result_dir = '/cache/result'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)         
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)       
    
    ###拷贝数据集到训练环境
    #DatasetToEnv(args.multi_data_url, data_dir)

    ###拷贝预训练模型文件到训练环境
    #pretrain_to_env(args.pretrain_url, pretrain_dir)

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    context.set_context(graph_kernel_flags="--opt_level=2 --dump_as_text")
    network = MSLT()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Adam(network.trainable_params(), learning_rate=cfg.lr, beta1=0.9, beta2=0.999)
    model = Model(network, net_loss, net_opt)

    print("============== Starting Testing ==============")

    param_dict = load_checkpoint("checkpoint_mslt-iter70000.ckpt")
    load_param_into_net(network, param_dict)
    ds_test = create_dataset(image_path="/dataset/valid/input",label_path="/dataset/valid/label", batch_size=1).create_dict_iterator()
    save_path="/dataset/valid/output/"
    if not os.path.exists(save_path):
    # 如果不存在，则创建文件夹
        os.makedirs(save_path)
    i = 1
    while True:
        data = next(ds_test)
        images = data["image"].asnumpy()
        labels = data["label"].asnumpy()
        print('Tensor:', Tensor(data['image']))
        output = model.predict(Tensor(data['image']))
        print('output:', output)
        output = output.squeeze(0)
        output = output.asnumpy()
        output = np.swapaxes(output, 1, 2) # CWH
        output = np.swapaxes(output, 0, 2)
        output = Image.fromarray(np.uint8(output*255))
        output.save(save_path+str(i)+".jpg")
        i+=1
    ###上传训练结果到启智平台  
    #env_to_openi(result_dir, args.result_url)