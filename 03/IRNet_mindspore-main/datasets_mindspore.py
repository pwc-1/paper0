"""
using args select dataset

"""

from dataset.HDRTV_set import HDRTV_set_mindspore



def Datasets(args=None,train=True,cfg=None):
    
    dataset = HDRTV_set_mindspore.Dataset(args=args, dataset_train=train,cfg=cfg)
    
    return dataset
