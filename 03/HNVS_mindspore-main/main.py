#!/usr/bin/env python
# coding: utf-8
#import torch
import numpy as np
import os
import zipfile

root_dir = './datasets/python'

img_list = np.load(os.path.join(root_dir, 'omniglot.npy')) # (1623, 20, 1, 28, 28)
x_train = img_list[:1200]
x_test = img_list[1200:]
num_classes = img_list.shape[0]
datasets = {'train': x_train, 'test': x_test}

### 准备数据迭代器
n_way = 5   # num_classes
k_spt = 1  ## support data 的个数 num_shot_train
k_query = 1 ## query data 的个数 num_shot_val
imgsz = 28
resize = imgsz
task_num = 32
batch_size = task_num

indexes = {"train": 0, "test": 0}
datasets = {"train": x_train, "test": x_test}
print("DB: train", x_train.shape, "test", x_test.shape)

def load_data_cache(dataset):
    """
    Collects several batches data for N-shot learning
    :param dataset: [cls_num, 20, 84, 84, 1]
    :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    #  take 5 way 1 shot as example: 5 * 1
    setsz = k_spt * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    for sample in range(10):  # num of epochs

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(batch_size):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], n_way, replace =  False) 

            for j, cur_class in enumerate(selected_cls):

                selected_img = np.random.choice(20, k_spt + k_query, replace = False)

                # 构造support集和query集
                x_spt.append(dataset[cur_class][selected_img[:k_spt]])
                x_qry.append(dataset[cur_class][selected_img[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_spt)
            x_spt = np.array(x_spt).reshape(n_way * k_spt, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_spt)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]
 
            # append [sptsz, 1, 84, 84] => [batch_size, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

#         print(x_spts[0].shape)
        # [b, setsz = n_way * k_spt, 1, 84, 84]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(np.int32).reshape(batch_size, setsz)
        # [b, qrysz = n_way * k_query, 1, 84, 84]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(np.int32).reshape(batch_size, querysz)
#         print(x_qrys.shape)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache

datasets_cache = {"train": load_data_cache(x_train),  # current epoch data cached
                       "test": load_data_cache(x_test)}

def next(mode='train'):
    """
    Gets next batch from the dataset with name.
    :param mode: The name of the splitting (one of "train", "val", "test")
    :return:
    """
    # update cache if indexes is larger than len(data_cache)
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1

    return next_batch


## omniglot

from models.adaCNN import adaCNNModel
import time
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE,device_target='CPU')
meta = adaCNNModel("model", num_classes=5)
meta_test = adaCNNModel("model", num_classes=5)
# meta = MetaLearner().to(device)

epochs = 60000
from mindspore import Tensor
import mindspore
from mindspore import nn
from mindspore import ops

loss_fn = nn.SoftmaxCrossEntropyWithLogits()
optimizer = nn.Adam(meta.trainable_params(),0.0003)
# Define forward 
def forward_fn(data, label, is_train):
    if is_train:
        logits,pre_acc = meta(data)
    else:
        logits,pre_acc = meta_test(data)
    softmax_cross = ops.SoftmaxCrossEntropyWithLogits()
    loss = softmax_cross(logits, label)  # logits:(160,5),label(160,5) 
    return loss, logits, pre_acc
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
def train_step(data, label, is_train=True):
    # b = data['train_labels'].shape[0]
    # for i in range(b):
    #     data_i = {
    #         "train_inputs": ops.expand_dims(data["train_inputs"][i], axis=0), # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
    #         "train_labels": ops.expand_dims(data["train_labels"][i], axis=0), # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
    #         "test_inputs": ops.expand_dims(data["test_inputs"][i], axis=0), # batch_size, num_classes * update_batch_size, 28 * 28
    #         "test_labels": ops.expand_dims(data["test_labels"][i], axis=0), # batch_size, num_classes * update_batch_size, num_classes
    #         }
    #     label_i = label[i*5:(i+1)*5]
    #     (loss, _, pre_acc), grads = grad_fn(data_i, label_i, is_train)
    #     optimizer(grads)
    (loss, _, pre_acc), grads = grad_fn(data, label, is_train)
    optimizer(grads)
    return loss, pre_acc



x_spt, y_spt, x_qry, y_qry = next('train')
x_spt = Tensor(x_spt)  # (32, 5, 1, 28, 28)
y_spt = Tensor(y_spt)      #(32, 5)
x_qry = Tensor(x_qry)   #(32, 75, 1, 28, 28)
y_qry = Tensor(y_qry)   #(32, 75)
one_hot_op = ops.OneHot()
depth, on_value, off_value = 5, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
y_spt= one_hot_op(y_spt,depth, on_value, off_value)
y_qry =  one_hot_op(y_qry,depth, on_value, off_value)
metatrain_input_tensors = {
"train_inputs": x_spt, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
"train_labels": y_spt, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
"test_inputs": x_qry, # batch_size, num_classes * update_batch_size, 28 * 28
"test_labels": y_qry, # batch_size, num_classes * update_batch_size, num_classes
}

x_spt, y_spt, x_qry, y_qry = next('test')
x_spt = Tensor(x_spt)  # (32, 5, 1, 28, 28)
y_spt = Tensor(y_spt)      #(32, 5)
x_qry = Tensor(x_qry)   #(32, 75, 1, 28, 28)
y_qry = Tensor(y_qry)   #(32, 75)
one_hot_op = ops.OneHot()
depth, on_value, off_value = 5, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
y_spt= one_hot_op(y_spt,depth, on_value, off_value)
y_qry =  one_hot_op(y_qry,depth, on_value, off_value)
metatest_input_tensors = {
"train_inputs": x_spt, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
"train_labels": y_spt, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
"test_inputs": x_qry, # batch_size, num_classes * update_batch_size, 28 * 28
"test_labels": y_qry, # batch_size, num_classes * update_batch_size, num_classes
}

for step in range(epochs):
    start = time.time()
    metatrain_iterations = 40000

    meta.set_train()

    metatrain_loss, pre_acc = train_step(metatrain_input_tensors, metatrain_input_tensors["test_labels"].view(-1, 5))
        
    if (step % 10) == 0:
        logits, train_accuracy = meta(metatrain_input_tensors)
        # print("logits arg:",logits.argmax(axis=1))
        # print("labels arg:",metatest_input_tensors["test_labels"].argmax(axis=2))
        test_labels = ops.reshape(metatrain_input_tensors["test_labels"].argmax(axis=2), (-1,))
        post_acc = (logits.argmax(axis=1) == test_labels).sum()*1.0/len(test_labels)
        print("step:", step)
        print("pre_acc:", train_accuracy)
        print("post_acc:", post_acc)
        print("train_loss:", metatrain_loss[0].mean())

    if (step+1) % 100 == 0:
        accs = 0
        idx = 0
        import copy
        meta_test = meta# copy.deepcopy(meta)
        train_step(metatest_input_tensors, metatest_input_tensors["test_labels"].view(-1, 5),False)
        logits, train_accuracy = meta_test(metatest_input_tensors)
        print("Test:")
        # print("logits:",logits)
        # print("labels:",metatest_input_tensors["test_labels"])
        # print("logits arg:",logits.argmax(axis=1))
        # print("labels arg:",metatest_input_tensors["test_labels"].argmax(axis=2))
        test_labels = ops.reshape(metatest_input_tensors["test_labels"].argmax(axis=2), (-1,))
        post_acc = (logits.argmax(axis=1) == test_labels).sum()*1.0/len(test_labels)
        print("pre_acc:", train_accuracy)
        print("Acc:", post_acc)