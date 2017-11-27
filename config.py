#coding:utf8


class Config:
    caption_data_path='caption.pth'
    img_path='/data/image/ai_cha/caption/ai_challenger_caption_train_20170902/caption_train_images_20170902'
    # img_path=
    feats_path = 'train_results.pth'
    scale_size = 256
    img_size = 256
    batch_size=32
    shuffle = True
    num_workers = 4
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights=False

    prefix='checkpoints/caption'#模型保存前缀

    env = 'caption'
    plot_every = 10
    debug_file = '/tmp/debc'

    model_ckpt = None # 模型断点保存路径
    lr1=1e-4
    lr2=1e-4
    use_gpu=True
    epoch = 2000

    test_img = 'test.jpg' 