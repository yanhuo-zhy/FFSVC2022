import sys
import logging
import argparse
import traceback
import pandas as pd
import tqdm
import soundfile
import os
import torch
import numpy
import torch.nn.functional as F
from model import ECAPA_TDNN
# Logger
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parse
def get_args():
    """Score:
            <key1, key2, score>
    Recommend: subset 2000 ~ 3000 utts from trainset as a cohort set and use asnorm with top-n=200 ~ 400.
    """
    parser = argparse.ArgumentParser(
            description="Score Normalization.")

    parser.add_argument("--method", default="asnorm", type=str,
                        choices=["snorm", "asnorm"],
                        help="Choices to select a score normalization.")

    parser.add_argument("--top-n", type=int, default=300,
                        help="Used in AS-Norm.")
    
    parser.add_argument("--second-cohort", type=str, default="true", choices=["true", "false"],
                        help="If true, get cohort key from the second field of score.")

    parser.add_argument("--cross-select", type=str, default="false", choices=["true", "false"],
                        help="Used in AS-Norm. "
                             "If true, select top n enroll/test keys by test/enroll_cohort scores. "
                             "If false, select top n enroll/test keys by enroll/test_cohort scores.")

    parser.add_argument("input_score", metavar="enroll-test-score", type=str,
                        help="Original score path for <enroll, test>.")

    parser.add_argument("enroll_cohort_score", metavar="enroll-cohort-score", type=str,
                        help="Score file path for <enroll, cohort>.")

    parser.add_argument("test_cohort_score", metavar="test-cohort-score", type=str,
                        help="Score file path for <test, cohort>.")

    parser.add_argument("output_score", metavar="output-score-path", type=str,
                        help="Output score path for <enroll, test> after score normalization.")

    args = parser.parse_args()

    return args

def load_score(score_path, names, sep=" "):
    logger.info("Load score form {} ...".format(score_path))
    df = pd.read_csv(score_path, sep=sep, names=names)
    return df

def save_score(score, score_path, sep=" "):
    logger.info("Save score to {} ...".format(score_path))
    df = pd.DataFrame(score)
    df.to_csv(score_path, header=None, sep=sep, index=False)

## 对称归一化
## 输入内容：
## 2000 - 3000条目标无关的说话人embedding，分别和注册语音和测试语音的得分。
## 存放到csv文件中，
## 文件1：注册语音和测试语音之间的得分
## 文件2：注册语音和冒认者集合语音之间的得分
## 文件3：测试语音和冒认者集合语音之间的得分
## 输出内容：
## 分数归一化的注册语音和测试语音之间的得分
## 
def snorm(args):
    """ Symmetrical Normalization.
    Reference: Kenny, P. (2010). Bayesian speaker verification with heavy-tailed priors. Paper presented at the Odyssey.
    """

    enroll_test_names = ["enroll", "test", "score"]

    if args.second_cohort == "true":
        enroll_cohort_names = ["enroll", "cohort", "score"]
        test_cohort_names = ["test", "cohort", "score"]
    else:
        enroll_cohort_names = ["cohort", "enroll", "score"]
        test_cohort_names = ["cohort", "test", "score"]


    input_score = load_score(args.input_score, enroll_test_names)
    enroll_cohort_score = load_score(args.enroll_cohort_score, enroll_cohort_names)
    test_cohort_score = load_score(args.test_cohort_score, test_cohort_names)

    output_score = []

    logger.info("Use Symmetrical Normalization (S-Norm) to normalize scores ...")

    # This .groupby function is really an efficient method than 'for' grammar.
    enroll_group = enroll_cohort_score.groupby("enroll")
    test_group = test_cohort_score.groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                       (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    logger.info("Normalize scores done.")
    save_score(output_score, args.output_score)


## 自适应的对称分数归一化（as-norm）
## 输入：
##  2000 - 3000条目标无关的说话人embedding，分别和注册语音和测试语音的得分。
## 存放到csv文件中，选择其中分数高的前300进行计算
## 文件1：注册语音和测试语音之间的得分
## 文件2：注册语音和冒认者集合语音之间的得分
## 文件3：测试语音和冒认者集合语音之间的得分
## 输出：
## 注册语音和测试语音之间的得分
##

def asnorm(args):
    """ Adaptive Symmetrical Normalization.
    Reference: Cumani, S., Batzu, P. D., Colibro, D., Vair, C., Laface, P., & Vasilakakis, V. (2011). Comparison of 
               speaker recognition approaches for real applications. Paper presented at the Twelfth Annual Conference 
               of the International Speech Communication Association.

               Cai, Danwei, et al. “The DKU-SMIIP System for NIST 2018 Speaker Recognition Evaluation.” Interspeech 2019, 
               2019, pp. 4370–4374.

    Recommend: Matejka, P., Novotný, O., Plchot, O., Burget, L., Sánchez, M. D., & Cernocký, J. (2017). Analysis of 
               Score Normalization in Multilingual Speaker Recognition. Paper presented at the Interspeech.

    """
    enroll_test_names = ["enroll", "test", "score"]

    if args.second_cohort == "true":
        enroll_cohort_names = ["enroll", "cohort", "score"]
        test_cohort_names = ["test", "cohort", "score"]
    else:
        enroll_cohort_names = ["cohort", "enroll", "score"]
        test_cohort_names = ["cohort", "test", "score"]

    input_score = load_score(args.input_score, enroll_test_names)
    enroll_cohort_score = load_score(args.enroll_cohort_score, enroll_cohort_names)
    test_cohort_score = load_score(args.test_cohort_score, test_cohort_names)

    output_score = []

    logger.info("Use Adaptive Symmetrical Normalization (AS-Norm) to normalize scores ...")

    # Note that, .sort_values function will return NoneType with inplace=True and .head function will return a DataFrame object.
    # The order sort->groupby is equal to groupby->sort, so there is no problem about independence of trials.
    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)

    if args.cross_select == "true":
        logger.info("Select top n scores by cross method.")
        # The SQL grammar is used to implement the cross selection based on pandas.
        # Let A is enroll_test table, B is enroll_cohort table and C is test_cohort table.
        # To get a test_group (select "test:cohort" pairs) where the cohort utterances' scores is selected by enroll_top_n,
        # we should get the D table by concatenating AxC with "enroll" key firstly and then
        # we could get the target E table by concatenating BxD wiht "test"&"cohort" key.
        # Finally, the E table should be grouped by "enroll"&"test" key to make sure the group key is unique.
        enroll_top_n = enroll_cohort_score.groupby("enroll").head(args.top_n)[["enroll", "cohort"]]
        test_group = pd.merge(pd.merge(input_score[["enroll", "test"]], enroll_top_n, on="enroll"), 
                              test_cohort_score, on=["test", "cohort"]).groupby(["enroll", "test"])

        test_top_n = test_cohort_score.groupby("test").head(args.top_n)[["test", "cohort"]]
        enroll_group = pd.merge(pd.merge(input_score[["enroll", "test"]], test_top_n, on="test"), 
                                enroll_cohort_score, on=["enroll", "cohort"]).groupby(["enroll", "test"])
    else:
        enroll_group = enroll_cohort_score.groupby("enroll").head(args.top_n).groupby("enroll")
        test_group = test_cohort_score.groupby("test").head(args.top_n).groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        if args.cross_select == "true":
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] + \
                                 (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                                (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    logger.info("Normalize scores done.")
    save_score(output_score, args.output_score)

def get_score(model, eval_list, eval_path):
    model.eval()
    model.to('cuda:0')
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()
    for line in lines:
        files.append(line.split()[0])
        files.append(line.split()[1])
    setfiles = list(set(files))
    setfiles.sort()
    print(len(setfiles))

    # ECAPA验证集处理
    for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
        audio, _  = soundfile.read(os.path.join(eval_path, file))
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
            embedding_1 = model.forward(data_1, aug = False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = model.forward(data_2, aug = False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings[file] = [embedding_1, embedding_2]
    scores = []

    # ECAPA验证集处理
    for line in lines:			
        embedding_11, embedding_12 = embeddings[line.split()[0]]
        embedding_21, embedding_22 = embeddings[line.split()[1]]
        # Compute the scores
        score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        score = (score_1 + score_2) / 2
        score = score.detach().cpu().numpy()
        scores.append(score)       

    return scores

def main():
    
    ## 分数归一化，输出参数s-norm or as-norm
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # print(" ".join(sys.argv))
    # args = get_args()
    # try:
    #     if args.method == "snorm":
    #         snorm(args)
    #     elif args.method == "asnorm":
    #         asnorm(args)
    #     else:
    #         raise TypeError("Do not support {} score normalization.".format(args.method))
    # except BaseException as e:
    #     if not isinstance(e, KeyboardInterrupt):
    #         traceback.print_exc()
    #     sys.exit(1)
    score = []
    checkpoint = torch.load("/home/zhy/FFSVC/exps/exp_baseline1/model_v1.pkl")
    model = ECAPA_TDNN(C=1024)
    model.load_state_dict(checkpoint['model'])
    score = get_score(model,'/home/wldong/ffsvc/dev_test/trials_eval','/home/wldong/ffsvc/dev_test/eval')
    save_score(score, '/home/zhy/FFSVC/exps/exp_baseline1')
    print('write score to log')
    with open('/home/zhy/FFSVC/scores.txt', 'a') as logs:
        logs.writelines([str(x)+'\n' for x in score])

if __name__ == "__main__":
    main()
# import os,tqdm
# from tkinter import scrolledtext
# import torch
# import numpy as np
# from utils import get_eer, get_score
# from ECAPAModel import ECAPAModel
# from model import ECAPA_TDNN
# from dataLoader import evaluate_loader
# from torch.utils.data import DataLoader

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# def validate(model,val_dataloader, val_file):
#     model.eval()
#     embd_dict={}
#     model.cuda()
#     with torch.no_grad():
#         for j, (feat, file_name) in tqdm.tqdm(enumerate(val_dataloader)):
#             outputs = model.forward(feat.cuda(), aug=False)  
#             for i in range(len(file_name)):
#                 embd_dict[file_name[i]] = outputs[i,:].cpu().numpy()
#     score = get_score(embd_dict, val_file)
#     # np.save('exp/%s/test_%s.npy' % (opt.save_dir, epoch),embd_dict)
#     return score

# def main():
#     # validation dataset
#     val_dataset = evaluate_loader('/home/wldong/ffsvc/dev_test/eval', 200)
#     val_dataloader = DataLoader(val_dataset, num_workers=8, pin_memory=True, batch_size=1)

#     # ECAPAModel
#     # model_00 --> 8.2 
#     # model_a0: 7.71 // model_a1: 7.43 // model_a2: 7.45  // model_a3: 7.73 // model_a4: 7.49 // model_a5: 7.42 // model_a6:7.075
#     # model_a7: 7.25 // model_a8: 7.44 // model_a9: 7.32 //
    

#     # pretrain_model = ECAPAModel(0.01, 0.97, 1024, 5994, 0.2, 30, 1)
#     # pretrain_model.load_parameters('/home/zhy/FFSVC/exps/exp_baseline1/model_f%d.pkl'%num)
#     # ecapa_tdnn = ECAPA_TDNN(C=1024)
#     # ecapa_tdnn = pretrain_model.speaker_encoder
#     print('load model form /home/zhy/FFSVC/exps/exp_baseline1/model_v1.pkl')
#     checkpoint = torch.load('/home/zhy/FFSVC/exps/exp_baseline1/model_v1.pkl')
#     ecapa_tdnn = ECAPA_TDNN(C=1024)
#     ecapa_tdnn.load_state_dict(checkpoint['model'])

#     # validation
#     score = validate(ecapa_tdnn, val_dataloader, "/home/wldong/ffsvc/dev_test/trials_eval")

#     print('write score to log')
#     with open('/home/zhy/FFSVC/score.txt', 'a') as logs:
#         logs.writelines([str(x)+'\n' for x in score])



# if __name__ == '__main__':
#     main()