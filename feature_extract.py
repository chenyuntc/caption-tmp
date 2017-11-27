#coding:utf8
'''
利用resnet50提取图片的语义信息
并保存层results.pth
'''
from config  import Config
import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image
import numpy as np

opt = Config()

IMAGENET_MEAN =  [0.485, 0.456, 0.406]
IMAGENET_STD =  [0.229, 0.224, 0.225]
normalize =  tv.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)

def fit_tensor(tensor):
    size = 3,320,320
    result = t.zeros(*size)
    _,h,w = tuple(tensor.size())
    result[:,320/2 -h/2:320/2+h/2, 320/2-w/2:320/2+w/2] = tensor
    return result

opt.train_img_path = '/data/image/ai_cha/caption/ai_challenger_caption_train_20170902/caption_train_images_20170902'
opt.val_img_path = '/data/image/ai_cha/caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910'

opt.batch_size=256
opt.num_workers=8


class CaptionDataset(data.Dataset):
    
    def __init__(self,caption_data_path):
        self.transforms = tv.transforms.Compose([
                tv.transforms.Scale(320),
                tv.transforms.CenterCrop(320),
                tv.transforms.ToTensor(),
                fit_tensor,
                normalize
        ])
    
        self.data = t.load(caption_data_path)
        self.train()


    def train(self):
        self.ix2id = self.data['train']['ix2id']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.train_img_path,self.ix2id[_]) \
                                for _ in range(len(self.ix2id))]

    def val(self):
        self.ix2id = self.data['val']['ix2id']
        self.imgs = [os.path.join(opt.val_img_path,self.ix2id[_]) \
                                for _ in range(len(self.ix2id))]

    def __getitem__(self,index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        return img,index

    def __len__(self):
        return len(self.imgs)
def main():

    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = data.DataLoader(dataset,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.num_workers,
                    )


    # 数据
    results = t.Tensor(len(dataloader.dataset),2048).fill_(0)
    batch_size = opt.batch_size
    dataset.train()

    # 模型
    resnet50 = tv.models.resnet50(pretrained=True)
    del resnet50.fc
    resnet50.fc = lambda x:x
    del resnet50.avgpool
    resnet50.avgpool = t.nn.AdaptiveAvgPool2d(1)
    resnet50.cuda()
    resnet50.eval()

    # 前向传播，计算分数
    for ii,(imgs, indexs)  in tqdm.tqdm(enumerate(dataloader)):
        # 确保序号没有对应错
        assert indexs[0]==batch_size*ii
        imgs = imgs.cuda()
        imgs = Variable(imgs,volatile=True)
        features = resnet50(imgs)
        results[ii*batch_size:(ii+1)*batch_size]= features.data.cpu()
    t.save(results,'train_results2.pth') 

    dataset = CaptionDataset(opt.caption_data_path)
    dataset.eval()
    dataloader = data.DataLoader(dataset,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.num_workers,
                    )

    results = t.Tensor(len(dataloader.dataset),2048).fill_(0)
    # 前向传播，计算分数
    for ii,(imgs, indexs)  in tqdm.tqdm(enumerate(dataloader)):
        # 确保序号没有对应错
        assert indexs[0]==batch_size*ii
        imgs = imgs.cuda()
        imgs = Variable(imgs,volatile=True)
        features = resnet50(imgs)
        results[ii*batch_size:(ii+1)*batch_size]= features.data.cpu()
    t.save(results,'val_results2.pth') 

def test2(**kwargs):
    class Dataset:
        def __init__(self):
            self.file_path = '/data/image/ai_cha/caption/ai_challenger_caption_test1_20170923/'
            self.imgs = t.load(self.file_path + 'test_imgs.pth')
            self.transforms = tv.transforms.Compose([
                tv.transforms.Scale(320),
                tv.transforms.CenterCrop(320),
                tv.transforms.ToTensor(),
                fit_tensor,
                normalize
        ])
        def __getitem__(self,index):
            img = Image.open(self.file_path+'caption_test1_images_20170923/'+self.imgs[index])
            img = self.transforms(img)
            return img
        
        def __len__(self):
            return len(self.imgs)

    dataset = Dataset()
    dataloader = data.DataLoader(dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                )
    # 模型
    resnet50 = tv.models.resnet50(pretrained=True)
    del resnet50.fc
    resnet50.fc = lambda x:x
    del resnet50.avgpool
    resnet50.avgpool = t.nn.AdaptiveAvgPool2d(1)
    resnet50.cuda()
    resnet50.eval()

    batch_size = opt.batch_size

    results = t.Tensor(len(dataloader.dataset),2048).fill_(0)
    # 前向传播，计算分数
    for ii,(imgs)  in tqdm.tqdm(enumerate(dataloader)):
        # 确保序号没有对应错
        # assert indexs[0]==batch_size*ii
        imgs = imgs.cuda()
        imgs = Variable(imgs,volatile=True)
        features = resnet50(imgs)
        results[ii*batch_size:(ii+1)*batch_size]= features.data.cpu()
    t.save(results,'test_results2.pth')

if __name__=='__main__':
    import fire
    fire.Fire()


