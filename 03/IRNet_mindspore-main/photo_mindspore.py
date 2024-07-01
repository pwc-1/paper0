import argparse
import mindspore
from mindspore.train import Model
import os
import cv2
import numpy as np
import argparse
from model.architecture_mindspore import IRNet_1, IRNet_2
from model.architecture_SRITM_mindspore import SRITM_IRNet_5


def traverse_under_folder(folder_root):
	folder_leaf = [] 
	folder_branch = []
	file_leaf = []
	
	index = 0
	for dirpath, subdirnames, filenames in os.walk(folder_root):
		index += 1
	
		if len(subdirnames) == 0:
			folder_leaf.append(dirpath)
		else:
			folder_branch.append(dirpath)
	
		for i in range(len(filenames)):
			file_leaf.append(os.path.join(dirpath, filenames[i]))

	return folder_leaf, folder_branch, file_leaf

testdata_path = '/home/csjunxu-3090/syb/HDRTV_test/test_sdr/'
_,_,fil = traverse_under_folder(testdata_path)
fil.sort()

mindspore.set_context(device_target='GPU')
mindspore.set_context(device_id=1)

model = IRNet_2(upscale=4)


param_dict = mindspore.load_checkpoint("/new/xlq/IRNet_mindspore/best.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
model.set_train(False)
model = Model(model)


for i in range(len(fil)):

    img_path = fil[i]
        
    img = cv2.cvtColor((cv2.imread(img_path,cv2.IMREAD_UNCHANGED) / 255).astype(np.float32), cv2.COLOR_BGR2YCrCb)
    img = mindspore.Tensor.from_numpy(np.transpose(img, (2, 0, 1))).float().clamp(min=0, max=1)
    img = mindspore.Tensor.unsqueeze(img,0)

    
    img = model.predict(img)



    img = mindspore.Tensor.squeeze(img,0).clamp(min=0, max=1)
    img = img.asnumpy()


    img = np.transpose(img,(1,2,0))
    img = img.astype(np.float32)
    img = ((img*65535).astype(np.uint16))
    
    
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)



    cv2.imwrite("./IRNet-2/results/"+str(i+1).rjust(3,'0')+".png", img)
    print(i)
