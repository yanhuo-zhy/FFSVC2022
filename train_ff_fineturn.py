import torch
import argparse, glob, os, warnings, time, tqdm, soundfile, sys
from tools import *
from dataLoader import train_loader, evaluate_loader
from ECAPAModel import ECAPAModel
from model import ECAPA_TDNN
from loss import AAMsoftmax
from utils import get_eer
from torch.nn import DataParallel

def get_pretrain(pretrain_path, n_class):
	# load pretrain ECAPAModel
	# pretrain_model = ECAPAModel(0.01, 0.97, 1024, 5994, 0.2, 30, 1)
	# pretrain_model.load_parameters(pretrain_path)
	checkpoint = torch.load(pretrain_path)
	print('Pretrain model loaded from %s '%pretrain_path)

	# get model
	model = ECAPA_TDNN(C=1024)
	# model = pretrain_model.speaker_encoder
	# n_class = n_class, s = 0.2, m = 30
	classifier = AAMsoftmax(n_class, 0.2, 30)

	model.load_state_dict(checkpoint['model'])
	classifier.load_state_dict(checkpoint['classifier'])

	# params need fineturn
	# layer4+relu2+attention+bn5+fc6+bn6 --> EER: 8.2 model_00
	# attention+bn5+fc6+bn6 --> 9.23
	# bn5+fc6+bn6 --> 8.9
	# fc6+nb6 -> 8.82 model_d0
	# fc6+nb6 -> 8.82 model_d0 && attention+bn5 -> 9.3   model_e0 
	# layer4+relu2+attention+bn5+fc6+bn6 --> EER: 8.2 model_00  && layer3  -->  8.01 model_f1  && layer2  --> 7.63 model_g2//7.60 model_g5
	# && layer1 -->7.48 model_h1 && conv1+relu1+bn1 -->7.35
	#--------------------------------------------------------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------------------------------------------------------
	# model_a6: 7.07 && attention + bn5 + fc6 + bn6  model_z0: 6.97 && layer4 + relu2 model_y0 : 6.725 && layer3 model_x1:6.55 && layer2 model_w1 : 6.33 && layer1 model_v1 : 6.20
	for name, param in model.named_parameters():
		param.requires_grad = ( name.startswith('layer1'))  #or name.startswith('bn1'))
	
	return model, classifier

def get_optim(model, classifier,hyperparameters):
	return torch.optim.Adam(
		   [param for param in model.parameters() if param.requires_grad]+list(classifier.parameters()),
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

def save_model(save_path, epoch, model, classifier, optimizer, scheduler):
    torch.save({'model': model.module.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }, os.path.join(save_path, 'model_ff_f%d.pkl' % epoch))

def train(model, train_dataloader, classifier, optimizer, scheduler, hyperparameters, epochs):
	model.train()
	classifier.train()

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
			outputs = model(feats, aug=True)
			nloss, prec = classifier(outputs, key)


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
		save_model("/home/zhy/FFSVC/exps/exp_ff", epoch, model, classifier, optimizer, scheduler)
		#eer, cost = validate(model, val_dataloader, "/home/wldong/ffsvc/dev_test/trials_dev_keys")
		#log.write('Epoch %d\t  lr %f\t  EER %.4f\t  cost %.4f\n'% (epoch, lr, eer*100, cost))


def main():
	warnings.simplefilter("ignore")
	hyperparameters = {
		'batch_size': 500*2,
		'num_workers': 32,
		'lr': 0.0001,
		'lr_decay': 0.97,
		'test_step': 1,
		'warmup_epoch': 2,
		'milestones': [15,30,40],
		'gamma': 0.1
	}
	# visible cuda
	os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
	# train dataset
	train_dataset = train_loader("/home/zhy/Data/list/train_ff.txt","/home/wldong/ffsvc",
					"/home/zhy/Data/musan","/home/wldong/ffsvc/rirs/RIRS_NOISES/simulated_rirs",
					200)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = hyperparameters['batch_size'], 
					   shuffle = True, num_workers = hyperparameters['num_workers'])
	# validation dataset
	# val_dataset = evaluate_loader('/home/wldong/ffsvc/dev_test/dev', 200)
	# val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=1)

	# get model & classifier /home/zhy/FFSVC/exps/exp_baseline1/model_a6.pkl   /home/zhy/FFSVC/pretrain.model
	model, classifier = get_pretrain("/home/zhy/FFSVC/exps/exp_ff/model_ff_e6.pkl",120)
	print(" Model para number = %.2f"%(sum(param.numel() for param in model.parameters() if param.requires_grad) / 1024 / 1024))
	# model = ECAPA_TDNN(C=1024)
	# classifier = AAMsoftmax(7325, 0.2, 30)
	# checkpoint = torch.load('/home/zhy/FFSVC/exps/exp_baseline/model/model_00.pkl')
	# model.load_state_dict(checkpoint['model'])
	# classifier.load_state_dict(checkpoint['classifier'])

	# get optimizer & scheduler
	optimizer = get_optim(model, classifier, hyperparameters)
	scheduler = get_scheduler(optimizer, hyperparameters)

	model.to('cuda:0')
	classifier.to('cuda:0')
	# dataparallel
	model = DataParallel(model)
	# classifier = DataParallel(classifier)

	# log
	# log = open('/home/zhy/FFSVC/exps/exp_baseline/score_fineturn.txt', 'w')
	# train
	print('start training!')
	train(model, train_dataloader, classifier, optimizer, scheduler, hyperparameters,10 )


if __name__ == '__main__':
    main()