'''
DataLoader for training
'''
import torch
import glob, numpy, os, random, soundfile, torch, time
from scipy import signal
from logmmse import logmmse_from_file
import librosa
# from nara_wpe.wpe import wpe
# from nara_wpe.utils import stft, istft

def mean_std_norm(signal):
	mean = torch.mean(signal).detach().data
	std  = torch.std(signal).detach().data
	std  = torch.max(std, torch.tensor(1e-6))
	return (signal - mean) / std

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		# 训练数据路径
		self.train_path = train_path
		# 输入端的持续时间，例如200代表持续2秒
		self.num_frames = num_frames
		# Load and configure augmentation files
		# musan噪声数据类型
		self.noisetypes = ['noise','speech','music']
		# 噪声信噪比
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		# 噪声数量
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		# 获取所有的噪声文件./data08/Others/musan_split/*/*/*.wav
		#             例如：./data08/Others/musan_split/music/fma/music-fma-0000.wav
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			#把musan数据集分成music,speech,noise三个字典，每个字典对应一个list，存放噪声文件完整路径名称
			self.noiselist[file.split('/')[-3]].append(file)
		# rir噪声文件list
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		# train_list记录格式&lines元素格式：id00012 id00012/21Uxsk56VDQ/00001.wav
		lines = open(train_list).read().splitlines()
		# dictkeys元素格式：id00012,用set去除重复元素
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		# key是'id00012', value是0
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			# speaker_label就是0,1,2,3,4...5994
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		#start_time = time.time()
		# Read the utterance and randomly select the segment
		# audio为音频数据,audio[0]是帧数,audio[1]是通道数，sr为采样频率(16KHz)
		audio, sr = soundfile.read(self.data_list[index])
		# length = 200*160+240 =32240 (16KHz的采样频率大概是2秒)
		# 有的音频文件不需要填充，有好几秒的长度		
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			# numpy.pad填充audio
			# （0，shortage）代表在数组前面填充0，在数组后面填充shortage个数值
			# 'wrap': 用原数组后面的值填充前面，前面的值填充后面
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		# random.random():随机小数
		# 随机截取audio
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		# audio增加维度，例：之前是（63488,0），现在是（1,63488）
		audio = numpy.stack([audio],axis=0)
		# Data Augmentation
		# augtype = random.randint(0,5)
		# if augtype == 0:   # Original
		# 	audio = audio
		# elif augtype == 1: # Reverberation
		# 	audio = self.add_rev(audio)
		# elif augtype == 2: # Babble
		# 	audio = self.add_noise(audio, 'speech')
		# elif augtype == 3: # Music
		# 	audio = self.add_noise(audio, 'music')
		# elif augtype == 4: # Noise
		# 	audio = self.add_noise(audio, 'noise')
		# elif augtype == 5: # Television noise
		# 	audio = self.add_noise(audio, 'speech')
		# 	audio = self.add_noise(audio, 'music')

		# 最终返回的是张量和对应的标签
		#end_time = time.time()
		#print('数据处理花费时间：',end_time-start_time)
		return mean_std_norm(torch.FloatTensor(audio[0])), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		# 从rir噪声数据集中随机选择一个
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		# rir增加维度，例：从（283504，）变为（1,283540）
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		# rir除以均方根
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		# 卷积audio和rir，"full':输出是输入的完全离散线性卷积。
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio


class evaluate_loader(object):
	def __init__(self, test_path, num_frames):
		self.test_path = test_path
		self.num_frames = num_frames
		self.test_list = glob.glob(self.test_path+'/*')


	def __len__(self):
		return len(self.test_list)


	def __getitem__(self, index):
		new_audio, sr = soundfile.read(self.test_list[index])
		# audio, sr = librosa.load(self.test_list[index],sr=None)
		# new_sample_rate = 16000
		# new_audio = librosa.resample(audio, orig_sr=sr, target_sr=new_sample_rate)
		# soundfile.write(self.test_list[index],new_audio,new_sample_rate)
		
		# audio = logmmse_from_file(self.test_list[index])	
		length = self.num_frames * 160 + 240
		if new_audio.shape[0] <= length:
			shortage = length - new_audio.shape[0]
			new_audio = numpy.pad(new_audio, (0, shortage), 'wrap')

		return mean_std_norm(torch.FloatTensor(new_audio)), self.test_list[index].split('/')[6]
