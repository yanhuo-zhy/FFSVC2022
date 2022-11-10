import torch
import argparse, glob, os, warnings, time, tqdm, soundfile, sys
from tools import *
from dataLoader import train_loader, evaluate_loader
from ECAPAModel import ECAPAModel
from model import ECAPA_TDNN
from loss import AAMsoftmax
from utils import get_eer
from torch.nn import DataParallel
from cross import CrossStitchNetwork

def get_pretrain(c_path, f_path, n_class):
    #获取近场模型
    # pretrain_model = ECAPAModel(0.01, 0.97, 1024, 5994, 0.2, 30, 1)
    # pretrain_model.load_parameters(c_path)
    # print('Pretrain model loaded from %s '%c_path)
    # model_c = pretrain_model.speaker_encoder
    checkpoint_c = torch.load(c_path)
    print('Pretrain model loaded from %s '%c_path)
    model_c = ECAPA_TDNN(C=1024)
    model_c.load_state_dict(checkpoint_c['model'])
    classifier_c = AAMsoftmax(n_class, 0.2, 30)
    classifier_c.load_state_dict(checkpoint_c['classifier'])
    #获取远场模型
    checkpoint_f = torch.load(f_path)
    print('Pretrain model loaded from %s '%f_path)
    model_f = ECAPA_TDNN(C=1024)
    model_f.load_state_dict(checkpoint_f['model'])
    # classifier_f = AAMsoftmax(n_class, 0.2, 30)
    # classifier_f.load_state_dict(checkpoint['classifier'])
    #分类器
    



    for name, param in model_f.named_parameters():
        param.requires_grad = ( name.startswith('None'))  #or name.startswith('bn1'))
    for name, param in model_c.named_parameters():
        param.requires_grad = ( name.startswith('conv1') or name.startswith('relu1')or name.startswith('bn1'))  #or name.startswith('bn1'))
    # 多任务模型
    cross_stitch_net = CrossStitchNetwork(model_c, model_f)

    return model_c, model_f, classifier_c, cross_stitch_net

def get_optim(cross_stitch_net, classifier_c, hyperparameters):
	return torch.optim.Adam(
		   [param for param in cross_stitch_net.parameters() if param.requires_grad]+list(classifier_c.parameters()),
    	   lr=hyperparameters['lr'], betas=(0.9, 0.999), 
    	   eps=1e-08, weight_decay=2e-5,
    	   amsgrad=False)
	

def get_scheduler(optimizer, hyperparameters,):
	return torch.optim.lr_scheduler.StepLR(optimizer, hyperparameters['test_step'], hyperparameters['lr_decay'])

def change_lr(optimizer,lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def validate(model,val_dataloader, val_file):
    model.eval()
    embd_dict={}
    with torch.no_grad():
        for j, (feat, file_name) in tqdm.tqdm(enumerate(val_dataloader)):
            outputs = model.forward(feat.cuda(), aug=False)  
            for i in range(len(file_name)):
                embd_dict[file_name[i]] = outputs[i,:].cpu().numpy()
    eer,_, cost,_ = get_eer(embd_dict, val_file)
    # np.save('exp/%s/test_%s.npy' % (opt.save_dir, epoch),embd_dict)
    return eer, cost

def save_model(save_path, epoch, cross_stitch_net, classifier_c):
    torch.save({'model': cross_stitch_net.state_dict(),
                'classifier': classifier_c.state_dict()
            }, os.path.join(save_path, 'model_fc_a%d.pkl' % epoch))


def train(cross_stitch_net, classifier_c, train_dataloader, optimizer, scheduler, hyperparameters, epochs):
    classifier_c.train()
    cross_stitch_net.train()

    lr_lambda = lambda x: hyperparameters['lr'] / (hyperparameters['batch_size'] * hyperparameters['warmup_epoch']) * (x + 1)
    for epoch in range(0,epochs):
        # start = time.time()
        scheduler.step(epoch)
        index, top1, loss = 0, 0, 0
        for i, (feats, key) in enumerate(train_dataloader, start = 1):
            optimizer.zero_grad()
            # if epoch < hyperparameters['warmup_epoch']:
                # change_lr(optimizer, lr_lambda(len(train_dataloader) * epoch + i))
            
            feats, key = feats.to('cuda:0'), torch.LongTensor(key).to('cuda:0')
            output_c, _ = cross_stitch_net(feats)             
            nloss, prec = classifier_c(output_c, key)

            nloss.backward()
            optimizer.step()

            index += len(key)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            lr = optimizer.param_groups[0]['lr']
            # batch_time = time.time() - start

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (i / train_dataloader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(i), top1/index*len(key)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        save_model("/home/zhy/FFSVC/exps/exp_ff", epoch, cross_stitch_net, classifier_c)
        #eer, cost = validate(model, val_dataloader, "/home/wldong/ffsvc/dev_test/trials_dev_keys")
        #log.write('Epoch %d\t  lr %f\t  EER %.4f\t  cost %.4f\n'% (epoch, lr, eer*100, cost))


def main():
    warnings.simplefilter("ignore")
    hyperparameters = {
        'batch_size': 200*2,
        'num_workers': 32,
        'lr': 0.00002,
        'lr_decay': 0.97,
        'test_step': 1,
        'warmup_epoch': 2,
        'milestones': [15,30,40],
        'gamma': 0.1
    }
    # visible cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
    # train dataset
    train_dataset = train_loader("/home/wldong/data/train_list_re.txt","/home/wldong/ffsvc",
                    "/home/zhy/Data/musan","/home/wldong/ffsvc/rirs/RIRS_NOISES/simulated_rirs",
                    200)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = hyperparameters['batch_size'], 
                        shuffle = True, num_workers = hyperparameters['num_workers'])
    # validation dataset
    # val_dataset = evaluate_loader('/home/wldong/ffsvc/dev_test/dev', 200)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=1)

    # get model & classifier /home/zhy/FFSVC/exps/exp_baseline1/model_a6.pkl   /home/zhy/FFSVC/pretrain.model
    model_c,  model_f, classifier_c, cross_stitch_net = get_pretrain("/home/zhy/FFSVC/exps/exp_baseline1/model_v1.pkl","/home/zhy/FFSVC/exps/exp_ff/model_ff_f8.pkl",7325)
    print(" Model para number = %.2f"%(sum(param.numel() for param in model_c.parameters() if param.requires_grad) / 1024 ))
    print(" Model para number = %.2f"%(sum(param.numel() for param in model_f.parameters() if param.requires_grad) / 1024 )) 
    print(" Model para number = %.2f"%(sum(param.numel() for param in cross_stitch_net.parameters() if param.requires_grad)/ 1024))   
    # model = ECAPA_TDNN(C=1024)
    # classifier = AAMsoftmax(7325, 0.2, 30)
    # checkpoint = torch.load('/home/zhy/FFSVC/exps/exp_baseline/model/model_00.pkl')
    # model.load_state_dict(checkpoint['model'])
    # classifier.load_state_dict(checkpoint['classifier'])

    # get optimizer & scheduler
    optimizer = get_optim(cross_stitch_net, classifier_c, hyperparameters)
    scheduler = get_scheduler(optimizer, hyperparameters)

    cross_stitch_net.to('cuda:0')
    classifier_c.to('cuda:0')
    # dataparallel
    cross_stitch_net = DataParallel(cross_stitch_net) 
    # classifier = DataParallel(classifier)

    # log
    # log = open('/home/zhy/FFSVC/exps/exp_baseline/score_fineturn.txt', 'w')
    # train
    print('start training!')
    train(cross_stitch_net, classifier_c, train_dataloader, optimizer, scheduler, hyperparameters,30)


if __name__ == '__main__':
    main()