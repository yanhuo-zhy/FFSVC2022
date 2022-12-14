'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''
#python自带的命令行参数解析包\通过通配符查找文件
import argparse, glob, os, torch, warnings, time, tqdm, soundfile
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from model import ECAPA_TDNN
start_time = time.time()
# 创建解析器对象
parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings 添加参数
# 输入端的持续时间，例如200代表持续2秒
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
# 最大epoch数值
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
# batch_size大小   
# 单卡batch size * 卡数200*2
parser.add_argument('--batch_size', type=int,   default=500,     help='Batch size')
# 
parser.add_argument('--n_cpu',      type=int,   default=16,       help='Number of loader threads')
# 每1个epoch测试和保存模型
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
# learning rate
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
# 每个epochlearning rate都衰减
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
# 训练数据list路径
parser.add_argument('--train_list', type=str,   default="/home/zhy/Data/list/train_ffsvc.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
# 训练数据路径
parser.add_argument('--train_path', type=str,   default="/home/wldong/ffsvc",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
# 测试数据list路径 /home/wldong/ffsvc/dev_test/trials_dev_keys /home/zhy/Data/list/trials_dev_partion.txt
parser.add_argument('--eval_list',  type=str,   default="/home/wldong/ffsvc/dev_test/trials_dev_keys",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# 测试数据路径
parser.add_argument('--eval_path',  type=str,   default="/home/wldong/ffsvc/dev_test/dev",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# musan噪声数据集路径
parser.add_argument('--musan_path', type=str,   default="/home/zhy/Data/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
# rir噪声数据集路径
parser.add_argument('--rir_path',   type=str,   default="/home/wldong/ffsvc/rirs/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
# 模型保存路径
parser.add_argument('--save_path',  type=str,   default="exps/expff1",                                     help='Path to save the score.txt and models')
# 初始化模型路径，默认为无  /home/zhy/ffsvc/ECAPA-TDNN-main/exps/pretrain.model
parser.add_argument('--initial_model',  type=str,   default="/home/zhy/FFSVC/exps/test/model_layer0_0.pkl",                                          help='Path of the initial_model')

## Model and Loss settings
# 说话人编码器的通道大小
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
# AAM softmax中的损失范围
# parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--m',       type=float, default=0.8,    help='Loss margin in AAM softmax')
# AAM softmax的损失规模
# parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
# voxceleb2说话人数量是5994 ffsvc是120
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')

## 测试模式Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
# multiprocessing模块用于在相同数据的不同进程中共享视图。file_system这个策略将提供文件名称给shm_open去定义共享内存区域。
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
# 在tool中模块定义，加入了模型和分数保存路径
args = init_args(args)

## train_loader定义了dataset，把dataset放入DataLoader中
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## 查找已经存在的模型文件
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## 如果在验证模式，必须要有初始化模型
if args.eval == True:
	s = ECAPAModel(**vars(args))
	#s = ECAPA_TDNN(C=args.C)
	# pg = [p for p in s.parameters() if p.requires_grad]
	# print("！！参数：", len(pg))
	print("Model %s loaded from previous state!"%args.initial_model)
	# s.load_parameters(args.initial_model)
	#checkpoint = torch.load(args.initial_model)
	model = ECAPA_TDNN(C=1024).to('cuda:0')
	model.load_parameters(args.initial_model)
	s.speaker_encoder = model
	# EER, minDCF = eval_network(s.to('cuda:4'), eval_list = args.eval_list, eval_path = args.eval_path)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))


EERs = []
score_file = open(args.score_save_path, "a+")
end_time = time.time()
print('训练前准备花费时间：',end_time-start_time)
while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1
