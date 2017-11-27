#coding:utf8
import os,ipdb
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torchnet import meter
import tqdm

from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from model import CaptionModel
from config import Config
from utils import Visualizer
from data import get_dataloader



def generate(**kwargs):
    opt = Config()
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    # 数据预处理
    data = t.load(opt.caption_data_path)
    word2ix,ix2word = data['word2ix'],data['ix2word']

    test_datas = t.load('test_results2.pth')
    imgs = t.load('test_imgs.pth')

    # Caption模型
    model = CaptionModel(opt,None,word2ix,ix2word)
    model = model.load(opt.model_ckpt).eval()
    model.cuda()

    results = []
    for ii,(img_feat,img_id) in tqdm.tqdm(enumerate(zip(test_datas,imgs))):
        sentences = model.generate(img_feat)
        item = {
            'image_id':img_id.replace('.jpg',''),
            'caption':sentences[0].replace('</EOS>','')
        }
        results.append(item)
        if ii%1000==0:print sentences[0]
    import json
    with open('submit.json','w') as f:
        json.dump(results,f)
    # results = model.generate(img_feats.data[0])
    # print('\r\n'.join(results))



def train(**kwargs):
    opt = Config()
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    vis = Visualizer(env = opt.env)
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix,ix2word = _data['word2ix'],_data['ix2word']
    
    # cnn = tv.models.resnet50(True)
    model = CaptionModel(opt,None,word2ix,ix2word)
    if opt.model_ckpt:
        model.load(opt.model_ckpt)
    
    optimizer = model.get_optimizer(opt.lr1)
    criterion = t.nn.CrossEntropyLoss()

    model.cuda()
    criterion.cuda()
    
    loss_meter = meter.AverageValueMeter()
    perplexity = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        
        loss_meter.reset()
        perplexity.reset()
        for ii,(imgs, (captions, lengths),indexes)  in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            input_captions = captions[:-1]
            imgs = imgs.cuda()
            captions = captions.cuda()

            imgs = Variable(imgs)
            captions = Variable(captions)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions,lengths)[0]
            
            score,_ = model(imgs,input_captions,lengths)
            loss = criterion(score,target_captions)
            loss.backward()
            # clip_grad_norm(model.rnn.parameters(),opt.grad_clip)
            optimizer.step()
            loss_meter.add(loss.data[0])
            perplexity.add(t.exp(loss.data)[0])

            # 可视化
            if (ii+1)%opt.plot_every ==0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
  
                vis.plot('loss',loss_meter.value()[0])
                vis.plot('perplexity',perplexity.value()[0])
                

                # 可视化原始图片


                raw_img = _data['train']['ix2id'][indexes[0]]
                img_path='/data/image/ai_cha/caption/ai_challenger_caption_train_20170902/caption_train_images_20170902/'+raw_img
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = tv.transforms.ToTensor()(raw_img)
                vis.img('raw',raw_img)

                # raw_img = (imgs.data[0]*0.25+0.45).clamp(max=1,min=0)
                # vis.img('raw',raw_img)
                
                # 可视化人工的描述语句
                raw_caption = captions.data[:,0]
                raw_caption = ''.join([_data['ix2word'][ii] for ii in raw_caption])
                vis.text(raw_caption,u'raw_caption')

                # 可视化网络生成的描述语句
                results = model.generate(imgs.data[0])
                vis.text('</br>'.join(results),u'caption')
        if (epoch+1)%100==0:
            model.save()

if __name__=='__main__':
    import fire
    fire.Fire()



