import os
from model_structure.MSLT import MSLT
from batch_transformers import BatchToTensor
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr
import cv2 as cv
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint
import mindspore as ms
from mindspore import nn
from mindspore.dataset.vision import py_transforms
from .dataset import *
import mindspore.dataset as ds

def train(config):
    ms.set_context(device_target="GPU")
    ms.set_context(device_id=1)
    model = MSLT()
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])  
    if config.load_pretrain == True:
        param_dict = load_checkpoint(config.pretrain_dir)
        load_param_into_net(model, param_dict)

    train_transform = py_transforms.ToTensor()
    valid_transform = BatchToTensor()
    
    
    l1_loss=nn.L1Loss(reduction='mean')
    loss_fn = nn.MSELoss(reduction='mean')

    train_dataset_generator = GetTrainDatasetGenerator(csv_file=os.path.join(config.datapath, 'test.txt'),
                                 img_dir=config.datapath,
                                 transform=train_transform)
    train_dataset = ds.GeneratorDataset(train_dataset_generator, ["train", "label"], shuffle=True)
    train_dataset = train_dataset.batch(batch_size=config.train_batch_size)

    valid_dataset_generator = dataset.GetValidDatasetGenerator(csv_file=os.path.join(config.validpath, 'test.txt'),
                                 Train_img_seq_dir=config.validpath,
                                 Label_img_dir=config.validlabel,
                                 Train_transform=valid_transform,
                                 Label_transform=py_transforms.ToTensor(),
                                 randomlist=False)
    
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator, ["train", "label"], shuffle=False)
    valid_dataset = valid_dataset.batch(batch_size=1)
    iters = len(train_dataset_generator)
    learning_rate = nn.cosine_decay_lr(1e-9, config.lr, iters*2, iters//config.train_batch_size+1, 50)
    optimizer = nn.Adam(model.trainable_params(), lr=learning_rate, betas=(0.9, 0.999))

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss


    for epoch in range(config.num_epochs):
        step = 0
        model.set_train()
        for sample_batched in train_dataset.create_dict_iterator():
            train_image, label_image = sample_batched['train'], sample_batched['label']
            loss = train_step(train_image, label_image)
            
            if ((step + 1) % config.display_iter) == 0:
                loss, current = loss.asnumpy(), step
                print(f"loss: {loss:>7f}  {current:>3d}")
            step += 1
        if ((epoch + 1) % config.snapshot_iter) == 0:
            ms.save_checkpoint(model, config.snapshots_folder + "Epoch" + str(epoch) + '.ckpt')
        if ((epoch + 1) % 2) == 0:
            time1 = 0
            count = 1
            evl1 = 0
            evl2 = 0
            evl_lpipsvgg = 0
            evl_lpipsalex = 0
            for sample_batched in valid_dataset.create_dict_iterator():
                train_image, label_image = sample_batched['train'], sample_batched['label']
                test_image = test_image.squeeze(0)
                print(label_image.shape)
                print(test_image.shape)

                label_image = label_image

                for index in range(5):
                    out4 = model(test_image[index].unsqueeze(0))
                    validate_path = "./validation/"
                    if not os.path.exists(validate_path):
                        os.makedirs(validate_path)

                    torchvision.utils.save_image(out4, validate_path + str(step + 1) + "_" + str(index + 1) + ".jpg")
                    image = cv.imread(validate_path + str(step + 1) + "_" + str(index + 1) + ".jpg")
                    label = cv.imread("../validation/label/" + str(step + 1) + ".jpg")

                    print(image.shape)
                    print(label.shape)

                    evl1 = evl1 + ssim(image, label, multichannel=True)
                    evl2 = evl2 + psnr(image, label)
                    evl_ssim = evl1 / (count)
                    evl_psnr = evl2 / (count)

                    count = count + 1

                    print("psnr", evl_psnr)
                    print("ssim", evl_ssim)
            f = "./valid_record/valid.txt"
            if not os.path.exists("./valid_record/"):
                os.makedirs("./valid_record/")
            with open(f, "a") as file:  # ”w"代表着每次运行都覆盖内容
                file.write("epoch="+str(epoch)+"_"+"ssim="+str(evl_ssim)+"psnr="+str(evl_psnr)+"\n")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--validpath', type=str, default="/dataset/validation/")
    parser.add_argument('--validlabel', type=str, default="/dataset/validation/label/")
    parser.add_argument('--datapath', type=str, default="/dataset/train_512/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    parser.add_argument('--grad_clip_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=199)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= 'snapshots131/Epoch131.pth')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    train(config)
