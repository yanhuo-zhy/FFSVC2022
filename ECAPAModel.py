'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle, time
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
# from logmmse import logmmse_from_file
# from nara_wpe.wpe import wpe
# from nara_wpe.utils import stft, istft

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'

class ECAPAModel(nn.Module):
	# 对应参数为：learning rate:0.01, learning rate decay: 0.97(每个epoch都衰减)，说话人编码器通道大小C：1024，说话人数量n_class：5994
	# Loss margin in AAM softmax m:0.2, Loss scale in AAM softmax s:30, Test and save every [test_step] epochs:1
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C)
		# self.speaker_encoder = nn.DataParallel(self.speaker_encoder, device_ids=[0,1,2])
		self.speaker_encoder.to('cuda:0')
		# # 指定要用到的设备
		# self.speaker_encoder = nn.DataParallel(ECAPA_TDNN(C = C), device_ids=device_ids)
		# # 模型加载到设备0
		# self.speaker_encoder = self.speaker_encoder.cuda(device=device_ids[0])
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).to('cuda:0')

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		# # 双卡并行
		# self.speaker_encoder = nn.DataParallel(self.speaker_encoder, device_ids=device_ids)
		# # 模型加载到设备0
		# self.speaker_encoder = self.speaker_encoder.cuda(device=device_ids[0])
		# self.speaker_loss.cuda(device=device_ids[0])
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			# labels            = torch.LongTensor(labels).cuda()
			labels            = torch.LongTensor(labels).to('cuda:0')
			# speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			speaker_embedding = self.speaker_encoder.forward(data.to('cuda:0'), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)	

			nloss.backward()
			self.optim.step()

			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		# # 模型加载到设备0
		# self.speaker_encoder = self.speaker_encoder.cuda(device=device_ids[0])
		# self.speaker_loss.cuda(device=device_ids[0])
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()
		print(len(setfiles))

		# ECAPA验证集处理
		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# # 此处加入声学前端处理********************************************
			# # 先进行logMMSE语音增强/降噪
			# audio = logmmse_from_file(os.path.join(eval_path, file))
			# # 再进行wpe去混响
			# audio = numpy.stack([audio], axis=0)
			# stft_options = dict(size=512, shift=128)
			# Y = stft(audio, **stft_options).transpose(2, 0, 1)
			# Z = wpe(Y)
			# audio = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
			# audio = numpy.squeeze(audio,axis=0)
			# # ***************************************************************
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to('cuda:0')

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).to('cuda:0')
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		# # Mine验证集处理
		# for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
		# 	audio, _ = soundfile.read(os.path.join(eval_path,file))
		# 	#音频长度处理
		# 	length = 200 * 160 + 240
		# 	if audio.shape[0] <= length:
		# 		shortage = length - audio.shape[0]
		# 		audio = numpy.pad(audio, (0,shortage), 'wrap')
		# 	#随机截取audio
		# 	start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		# 	audio = audio[start_frame:start_frame + length]
		# 	data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).to('cuda:0')
		# 	# speaker embeddings
		# 	with torch.no_grad():
		# 		embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
		# 		embedding_1 = F.normalize(embedding_1, p=2, dim=1)
		# 	embedings[file] = [embedding_1]
		# scores, labels = [], []

		# ECAPA验证集处理
		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))

		# # Mine验证集处理
		# for line in lines:
		# 	embedding_11 = embeddings[line.split()[1]]
		# 	embedding_21 = embeddings[line.split()[2]]
		# 	# compute the scores
		# 	score = torch.mean(torch.matmul(embedding_11,embedding_21.T))
		# 	score = score.detach().cpu().numpy()
		# 	scores.append(score)
		# 	labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)