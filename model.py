#coding:utf8
import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision as tv
from torchvision import transforms
from utils.beam_search import CaptionGenerator
import time
from itertools import chain

class CaptionModel(nn.Module):
    def __init__(self,opt,cnn,word2ix,ix2word):
        super(CaptionModel, self).__init__() 
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt
        
        self.fc = nn.Linear(2048,opt.rnn_hidden)
        self.rnn = nn.LSTM(opt.embedding_dim,opt.rnn_hidden,num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden,len(word2ix))
        self.embedding = nn.Embedding(len(word2ix),opt.embedding_dim)
        if opt.share_embedding_weights:
            # rnn_hidden=embedding_dim的时候才可以
            self.embedding.weight
        #### delete this now#####
        # self.load(opt.model_ckpt)
        # opt.model_ckpt=None
        # ######################
        # self.cnn = cnn.eval()
        # del cnn.fc
        # cnn.fc = lambda x:x

# ###########################################################
#         IMAGENET_MEAN =  [0.485, 0.456, 0.406]
#         IMAGENET_STD =  [0.229, 0.224, 0.225]
#         normalize =  tv.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
#         transforms = tv.transforms.Compose([
#                     tv.transforms.Scale(opt.scale_size),
#                     tv.transforms.CenterCrop(opt.img_size),
#                     tv.transforms.ToTensor(),
#                     normalize
#             ])
#         from PIL import Image
#         img = Image.open(opt.test_img)
#         img = transforms(img).unsqueeze(0)

#         # 用resnet50来提取图片特征
#         resnet50 = tv.models.resnet50(True).eval()
#         del resnet50.fc
#         resnet50.fc = lambda x:x
#         resnet50.cuda() 
#         img = img.cuda()
#         from torch.autograd import Variable
#         self.cuda()
#         print self.generate(img[0])[0]

#         img_feats = resnet50(Variable(img,volatile=True))
#         # img_feats = self.fc(img_feats)


#         eos_token='</EOS>'
#         beam_size=3
#         max_caption_length=20
#         length_normalization_factor=0.0
#         cap_gen = CaptionGenerator(embedder=self.embedding,
#                                    rnn=self.rnn,
#                                    classifier=self.classifier,
#                                    eos_id=self.word2ix[eos_token],
#                                    beam_size=beam_size,
#                                    max_caption_length=max_caption_length,
#                                    length_normalization_factor=length_normalization_factor)

#         img = t.autograd.Variable(img.unsqueeze(0), volatile=True)
#         img_feats = self.fc(img_feats).unsqueeze(0)
#         sentences, score = cap_gen.beam_search(img_feats)
#         sentences = [''.join([self.ix2word[idx] for idx in sent])
#                      for sent in sentences]
#         print sentences[0]

#         import ipdb
#         ipdb.set_trace()
################################################################33##########


        # cnn.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,imgs,captions,lengths):
        embeddings = self.embedding(captions)
        # img_feats = self.cnn(imgs)
        imgs = self.fc(imgs).unsqueeze(0)
        embeddings = t.cat([imgs,embeddings],0)
        # lengths？有问题不？
        packed_embeddings = pack_padded_sequence(embeddings,lengths)
        outputs,state = self.rnn(packed_embeddings)
        pred = self.classifier(outputs[0])
        return pred,state

    def generate(self, img, eos_token='</EOS>',
                 beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0):

        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img = t.autograd.Variable(img.unsqueeze(0), volatile=True)
        # img_feats = self.cnn(img)
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [''.join([self.ix2word[idx] for idx in sent])
                     for sent in sentences]
        return sentences

    def states(self):
        opt_state_dict = {attr:getattr(self.opt,attr) 
                                for attr in dir(self.opt) 
                                    if not attr.startswith('__') }
        return {
            'state_dict':self.state_dict(),
            'opt':opt_state_dict
        }


    def save(self,path=None,**kwargs):
        if path is None:
            path = '{prefix}_{time}'.format(prefix = self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        t.save(states, path)
        return path
        
    def load(self,path,load_opt=False):
        data = t.load(path)
        state_dict = data['state_dict']
        # for key in list(state_dict.keys()):
        #     if 'cnn' in key:
        #         del state_dict[key]
        self.load_state_dict(state_dict)
        
        if load_opt:
            for k,v in data['opt'].items():
                setattr(self.opt,k,v)

        return self
    
    # def get_optimizer(self,lr1,lr2):
    #     new_params = []
    #     new_params = chain( self.rnn.parameters(),
    #                         self.fc.parameters(),
    #                         self.classifier.parameters(),
    #                         self.embedding.parameters())

    #     new_params = {param for param in new_params}
    #     new_params_ids = {id(param) for param in new_params}
    #     pretrained_params = [param for param in self.cnn.parameters() if id(param) not in new_params_ids]
        
    #     param_groups = [
    #          {'params':pretrained_params,'lr':lr1},
    #                 {'params':new_params,'lr':lr2},
                   
    #             ]
    #     return t.optim.Adam(param_groups)
    def get_optimizer(self,lr):
        return t.optim.Adam(self.parameters(),lr=lr)