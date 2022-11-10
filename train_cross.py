'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''
#python自带的命令行参数解析包\通过通配符查找文件
import argparse, glob, os, warnings, time, sys, tqdm, soundfile
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from loss import AAMsoftmax
from model import ECAPA_TDNN
from cross import CrossStitchNetwork
import torch

# 创建预训练解析器对象
parser_pre = argparse.ArgumentParser(description = "ECAPA_pretrain_model")
parser_pre.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser_pre.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser_pre.add_argument('--batch_size', type=int,   default=200,     help='Batch size')
parser_pre.add_argument('--n_cpu',      type=int,   default=16,       help='Number of loader threads')
parser_pre.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser_pre.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser_pre.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
parser_pre.add_argument('--train_list', type=str,   default="/home/zhy/Data/list/train_ff.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser_pre.add_argument('--train_path', type=str,   default="/home/wldong/ffsvc",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser_pre.add_argument('--eval_list',  type=str,   default="/home/zhy/Data/list/trials_dev_partion.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser_pre.add_argument('--eval_path',  type=str,   default="/home/wldong/ffsvc/dev_test/dev",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser_pre.add_argument('--musan_path', type=str,   default="/home/zhy/Data/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser_pre.add_argument('--rir_path',   type=str,   default="/home/wldong/ffsvc/rirs/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser_pre.add_argument('--save_path',  type=str,   default="exps/exp_cross2",                                     help='Path to save the score.txt and models')
parser_pre.add_argument('--initial_model',  type=str,   default="/home/zhy/FFSVC/pretrain1.model",                                          help='Path of the initial_model')
parser_pre.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser_pre.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser_pre.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser_pre.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
parser_pre.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

# 创建解析器对象
parser = argparse.ArgumentParser(description = "ECAPA_trainer")
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=200,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=16,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
parser.add_argument('--train_list', type=str,   default="/home/zhy/Data/list/train_ff.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/home/wldong/ffsvc",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/home/zhy/Data/list/trials_dev_partion.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/home/wldong/ffsvc/dev_test/dev",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/home/zhy/Data/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/home/wldong/ffsvc/rirs/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="exps/exp_cross2",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="/home/zhy/FFSVC/pretrain1.model",                                          help='Path of the initial_model')
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.8,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=120,   help='Number of speakers')
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
# multiprocessing模块用于在相同数据的不同进程中共享视图。file_system这个策略将提供文件名称给shm_open去定义共享内存区域。
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args_pre = parser_pre.parse_args()
# 在tool中模块定义，加入了模型和分数保存路径
args = init_args(args)

## train_loader定义了dataset，把dataset放入DataLoader中
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## 查找已经存在的模型文件
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## 初始化模型
s = ECAPAModel(**vars(args_pre))
print("Model %s loaded from previous state!"%args.initial_model)
s.load_parameters(args.initial_model)
pretrain_model = s.speaker_encoder
## 远场模型
ecapa_model = ECAPA_TDNN(C=args.C)

# 多任务模型
cross_stitch_net = CrossStitchNetwork(ecapa_model, pretrain_model).to('cuda:0')

# 用于优化的参数
parameter_a = [param for param in ecapa_model.parameters()]
parameter_b = [param for param in cross_stitch_net.parameters()]
parameter_list = parameter_a+parameter_b


# speaker_loss
speaker_loss = AAMsoftmax(n_class = args.n_class, m = args.m, s = args.s).to('cuda:0')

# 优化器
optim = torch.optim.Adam(parameter_list, lr = args.lr, weight_decay = 2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = args.test_step, gamma = args.lr_decay)
print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in parameter_a) / 1024 / 1024))
print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in parameter_b) / 1024 / 1024))

# 训练
def train(cross_stitch_net, speaker_loss, epoch, loader, optim, scheduler):
    cross_stitch_net.train()
    scheduler.step(epoch-1)
    index, top1, loss = 0, 0, 0
    lr = optim.param_groups[0]['lr']
    for num, (data, labels) in enumerate(loader, start = 1):
        optim.zero_grad()
        labels = torch.LongTensor(labels).to('cuda:0')
        speaker_embedding, _ = cross_stitch_net.forward(data.to('cuda:0'))
        nloss, prec       = speaker_loss.forward(speaker_embedding, labels)
        nloss.backward()
        optim.step()

        index += len(labels)
        top1 += prec
        loss += nloss.detach().cpu().numpy()
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
        " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
        " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
        sys.stderr.flush()
    sys.stdout.write("\n")
    return loss/num, lr, top1/index*len(labels)

# 测试
def eval_network(model, eval_list, eval_path):
    model.eval()
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
            embedding_1 = model.forward(data_1, aug = False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = model.forward(data_2, aug = False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings[file] = [embedding_1, embedding_2]
    scores, labels  = [], []

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

    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF





if __name__ == '__main__':
    epoch = 1
    EERs = []
    score_file = open(args.score_save_path, "a+")

    while(1):
        ## Training for one epoch
        loss, lr, acc = train(cross_stitch_net,speaker_loss,epoch,trainLoader,optim,scheduler)

        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            ecapa_model.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
            EERs.append(eval_network(ecapa_model,eval_list = args.eval_list, eval_path = args.eval_path)[0])
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
