import mindspore
from mindspore.dataset import GeneratorDataset
from initial_environment_mindspore import init_env
from mindspore.amp import all_finite
import numpy as np
from models_mindspore import *
from datasets_mindspore import *
import argparse
import pandas as pd
from yacs.config import CfgNode as CN
from configs import *
from initializers_mindspore import *
from optimizers_mindspore import *
from schedulers_mindspore import *
import os


# args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default="SRITM",help="select dataset")
parser.add_argument('--model',type=str,default="IRNet-2",help="select model")
parser.add_argument('--shuffle',type=str,default=True,help=" train shuffle")
parser.add_argument('--epoch_base', type=int, default=0)
parser.add_argument('--pretrain_path',type=str,default=None)
parser.add_argument('--channels',type=int,default=64)
parser.add_argument('--last_max_psnr',type=float,default=0)
parser.add_argument('--last_max_ssim',type=float,default=0)
parser.add_argument('--run_eval',type=str,default=True)
args = parser.parse_args()


# save model path
if args.model == 'IRNet-2':
    save_path = "./"+'IRNet-2'+'/'+ args.dataset+"_"+args.model+"_"+str(args.shuffle)+".pt"
    dir_path = "./"+'IRNet-2'+'/'
    epoch_pre = 'IRNet-2'
else:
    raise NotImplementedError("not available")


# cfg
cfg_path = config_path(args=args)
cfg = CN.load_cfg(open(cfg_path))

init_env(cfg)
cfg.freeze()

# random seed
np.random.seed(0)
mindspore.set_seed(0)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


# dataset
train_set = Datasets(args=args,train=True,cfg=cfg)
valid_set = Datasets(args=args,train=False,cfg=cfg)

# 从下面开始改

# dataloader
train_dataloader = GeneratorDataset(train_set,column_names=['lq', 'gt'],sampler=None,shuffle=True,num_parallel_workers=2)
train_dataloader = train_dataloader.batch(batch_size=cfg.TRAIN_BATCH_SIZE)  # num_parallel_workers 不知道要不要加

valid_dataloader = GeneratorDataset(valid_set,column_names=['lq', 'gt'],sampler=None,shuffle=False,num_parallel_workers=2)
valid_dataloader = valid_dataloader.batch(batch_size=1)


# 初始化模型
model = Net(args,cfg=cfg)
print(model)
choose_initializer(cfg=cfg) 
model = init_weight(model)
# if args.pretrain_path is not None:
#     model = torch.load(args.pretrain_path).cuda()

# optimizer
optimizer = Optimizer(model=model,args=args,cfg=cfg)

#scheduler
scheduler = Scheduler(optimizer=optimizer,args=args,cfg=cfg)

epoch_base = args.epoch_base

mean_psnr_max = 0.0
mean_ssim_max = 0.0
print("mean_psnr_max:%.4f, mean_ssim_max:%.4f" %(mean_psnr_max,mean_ssim_max))
metrics_csv = {}
metrics_csv["psnr"] = []
metrics_csv["ssim"] = []

# train valid 
train_dataloader = train_dataloader.create_dict_iterator()
valid_dataloader = valid_dataloader.create_dict_iterator()
model.set_train(True)

def forward_fn(input, gt):
    output = model(input)
    loss = mindspore.ops.l1_loss(output, gt)
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(input, gt):
    loss, grads = grad_fn(input, gt)
    optimizer(grads)
    return loss

for epoch in range(epoch_base,cfg.MAX_EPOCH):
    # 训练一个epoch
    for batch, data in enumerate(train_dataloader):
        loss = train_step(data['lq'], data['gt']).asnumpy()
        
        print(f"epoch: [{epoch}], batch: [{batch}], "
                f"loss: {loss}", flush=True)
    
    if epoch % 10 ==0:
        epoch_save_path = "./"+epoch_pre +'/' + str(epoch)+'_'+args.model+".ckpt"
        mindspore.save_checkpoint(model, epoch_save_path)
    elif epoch == cfg.MAX_EPOCH - 1:
        epoch_save_path = "./"+epoch_pre +'/' + str(epoch)+'_'+args.model+".ckpt"
        mindspore.save_checkpoint(model, epoch_save_path)
    
    # param_dict = mindspore.load_checkpoint("/new/xlq/IRNet_mindspore/IRNet-2/0_IRNet-2.ckpt")
    # param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    
    
    # 推理并保存最好的那个checkpoint
    if args.run_eval:
        # mindspore.set_context(device_target='CPU')
        model.set_train(False)

        metrics = {}
        metrics['psnr'] = []
        metrics['ssim'] = []
        
        for batch, data in enumerate(valid_dataloader):
            output = model(data["lq"]).asnumpy()
            gt = data["gt"].asnumpy()
            
            metrics = valid_set.__measure__(output=output, gt=gt,metrics=metrics)  # 感觉这里要先把gt和output换成np
            
        mean_psnr = sum(metrics['psnr'])/len(metrics['psnr'])
        mean_ssim = sum(metrics['ssim'])/len(metrics['ssim'])
            
        print(f"epoch {epoch}, psnr: {mean_psnr}, ssim: {mean_ssim}", flush=True)
            
        if (mean_psnr >= mean_psnr_max) & (mean_ssim >= mean_ssim_max):
            mean_psnr_max = mean_psnr
            mean_ssim_max = mean_ssim

            mindspore.save_checkpoint(model, "best.ckpt")
            
            mt_max = pd.DataFrame({"psnr_max":[mean_psnr_max],"ssim_max":[mean_ssim_max]})
            mt_max.to_csv(dir_path+ args.dataset+"_"+args.model+"_"+str(args.shuffle)+'_metrics_max',index=False,sep=",")
            
            print(f"Updata best psnr: {mean_psnr}, ssim: {mean_ssim}")
        
        # mindspore.set_context(device_target='GPU')
        model.set_train(True)