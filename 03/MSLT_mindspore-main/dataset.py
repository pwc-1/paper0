# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
import os
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.transforms.py_transforms as py_transforms
from PIL import Image


class Dataset:
    def __init__(self, image_list, label_list,image_path,label_path):
        super(Dataset, self).__init__()
        self.imgs = image_list
        self.labels = label_list
        self.image_path = image_path
        self.label_path = label_path

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_path,self.imgs[index])).convert('RGB')
        label = Image.open(os.path.join(self.label_path,self.labels[index])).convert('RGB')
        return img, label
 
    def __len__(self):
        return len(self.imgs)
    
    
class MySampler():
    def __init__(self, dataset):
        self.__num_data = len(dataset)

    def __iter__(self):
        indices = list(range(self.__num_data))
        return iter(indices)


def create_dataset(image_path, label_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    save_image_list = os.listdir(image_path)
    save_label_list = os.listdir(label_path)
    dataset = Dataset(save_image_list, save_label_list,image_path,label_path)
    sampler = MySampler(dataset)
    me_ds = ds.GeneratorDataset(dataset, column_names=["image", "label"], sampler=sampler, shuffle=True)
    transforms_list = py_transforms.Compose([
            py_vision.ToTensor()])
    me_ds = me_ds.map(operations=transforms_list, input_columns="image")
    me_ds = me_ds.map(operations=transforms_list, input_columns="label")

#     # apply DatasetOps
    buffer_size = 10
    me_ds = me_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    me_ds = me_ds.batch(batch_size)
#     me_ds = me_ds.repeat(repeat_size)

    return me_ds
