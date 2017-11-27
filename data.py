#coding:utf8
import torch as t
from torch.utils import data
import os
from PIL import Image
import torchvision as tv
import numpy as np

IMAGENET_MEAN =  [0.485, 0.456, 0.406]
IMAGENET_STD =  [0.229, 0.224, 0.225]

# - 区分训练集和验证集
# - 不是随机返回每句话，而是根据index%5
# - 

# def create_collate_fn():
#     def collate_fn():
#         pass
#     return collate_fn

def create_collate_fn( padding,eos,max_length=50):
    def collate_fn(img_cap):
        '''
        将多个样本拼接在一起成一个batch
        输入： list of data，形如
        [(img1, cap1, index1), (img2, cap2, index2) ....]
        
        拼接策略如下：
        - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
          选取长度最长的句子，将所有句子pad成一样长
        - 长度不够的用</PAD>在结尾PAD
        - 没有START标识符
        - 如果长度刚好和词一样，那么就没有</EOS>
        
        返回：
        - imgs(Tensor): batch_sie*2048
        - cap_tensor(Tensor): batch_size*max_length
        - lengths(list of int): 长度为batch_size
        - index(list of int): 长度为batch_size
        '''
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps,indexes = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths),indexes)
    return collate_fn

class CaptionDatasetOLD(data.Dataset):
    
    def __init__(self,opt,transforms=None,training=True):
        self.opt = opt
        self.training=training
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.normalize =  tv.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
        self.train()
        


    def __getitem__(self,index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        caption = self.captions[index]
        rdn_index = np.random.choice(len(caption),1)[0]
        caption = caption[rdn_index]
        return img,t.LongTensor(caption)

        
    def __len__(self):
        return len(self.imgs)
        
    def train(self):
        '''
        trainging为True，返回训练集的数据，否则返回验证集的数据
        '''
        self.ix2id = self._data['train']['ix2id']
        self.imgs = [os.path.join(self.opt.img_path,self.ix2id[_]) for _ in range(len(self.ix2id))]
        self.captions = self._data['train']['caption']
        self.transforms =  tv.transforms.Compose([
            tv.transforms.Scale(self.opt.scale_size),
            tv.transforms.RandomCrop(self.opt.img_size),
            tv.transforms.ToTensor(),
            self.normalize
        ])     
        # self.transforms = tv.transforms.Compose([
        #         tv.transforms.Scale(opt.scale_size),
        #         tv.transforms.CenterCrop(opt.img_size),
        #         tv.transforms.ToTensor(),
        #         self.normalize
        # ])

        return self
    def val(self):
        '''
        trainging为True，返回训练集的数据，否则返回验证集的数据
        '''   
        self.ix2id = self._data['val']['ix2id']
        self.imgs = [os.path.join(opt.img_path,self.ix2id[_]) for _ in range(len(self.ix2id))]
        self.captions = self._data['val']['caption']

        self.transforms = tv.transforms.Compose([
                tv.transforms.Scale(self.opt.scale_size),
                tv.transforms.CenterCrop(self.opt.img_size),
                tv.transforms.ToTensor(),
                self.normalize
        ])
        return self



class CaptionDataset(data.Dataset):
    
    def __init__(self,opt):
        self.opt = opt
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.img_feats = t.load(opt.feats_path)
        self.train()
        
    def __getitem__(self,index):
        img = self.img_feats[index]
        caption = self.captions[index]
        rdn_index = np.random.choice(len(caption),1)[0]
        caption = caption[rdn_index]
        return img,t.LongTensor(caption),index

        
    def __len__(self):
        return len(self.img_feats)
        
    def train(self):
        '''
        trainging为True，返回训练集的数据，否则返回验证集的数据
        '''
        # self.img_feats= self.img_feats
        self.captions = self._data['train']['caption']
       

        return self
    def val(self):
        '''
        trainging为True，返回训练集的数据，否则返回验证集的数据
        '''   
        # self.img_feats = self.PP
        #!TODO: fix it
        self.captions = self._data['val']['caption']
        return self


def get_dataloader(opt,training=True):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    num_workers=opt.num_workers,
                    collate_fn=create_collate_fn(dataset.padding,dataset.end))
    return dataloader

if __name__=='__main__':
    from config import Config
    opt = Config()
    dataloader = get_dataloader(opt) 
    for ii,data in enumerate(dataloader):
        print ii,data
        break