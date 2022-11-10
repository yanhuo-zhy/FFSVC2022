import os,tqdm
import torch
import numpy as np
from utils import get_eer
from ECAPAModel import ECAPAModel
from model import ECAPA_TDNN
from dataLoader import evaluate_loader
from torch.utils.data import DataLoader
from cross import CrossStitchNetwork

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def validate(model,val_dataloader, val_file):
    model.eval()
    embd_dict={}
    model.cuda()
    with torch.no_grad():
        for j, (feat, file_name) in tqdm.tqdm(enumerate(val_dataloader)):
            outputs= model.forward(feat.cuda(), aug=False)  
            # outputs, _ = model.forward(feat.cuda())
            for i in range(len(file_name)):
                embd_dict[file_name[i]] = outputs[i,:].cpu().numpy()
    eer,_, cost,_ = get_eer(embd_dict, val_file)
    # np.save('exp/%s/test_%s.npy' % (opt.save_dir, epoch),embd_dict)
    return eer, cost

def main():
    # validation dataset
    val_dataset = evaluate_loader('/home/wldong/ffsvc/dev_test/dev', 200)
    val_dataloader = DataLoader(val_dataset, num_workers=8, pin_memory=True, batch_size=1)

    # ECAPAModel
    # model_00 --> 8.2 
    # model_a0: 7.71 // model_a1: 7.43 // model_a2: 7.45  // model_a3: 7.73 // model_a4: 7.49 // model_a5: 7.42 // model_a6:7.075
    # model_a7: 7.25 // model_a8: 7.44 // model_a9: 7.32 //
    

    # pretrain_model = ECAPAModel(0.01, 0.97, 1024, 5994, 0.2, 30, 1)
    # pretrain_model.load_parameters('/home/zhy/FFSVC/pretrain.model')
    # ecapa_tdnn = ECAPA_TDNN(C=1024)
    # ecapa_tdnn = pretrain_model.speaker_encoder
  
    # checkpoint = torch.load('/home/zhy/FFSVC/exps/test/model_layer0_0.pkl')
    # ecapa_tdnn = ECAPA_TDNN(C=1024)
    # ecapa_tdnn.load_state_dict(checkpoint['model'])
    ecapa_tdnn = ECAPA_TDNN(C=1024)
    ecapa_tdnn.load_parameters('/home/zhy/FFSVC/exps/test/model_layer0_11.pkl')
    print("Model %s loaded from previous state!"%'/home/zhy/FFSVC/exps/test/model_layer0_11.pkl')
    # checkpoint = torch.load('/home/zhy/FFSVC/exps/exp_best/model_best_a0.pkl')
    # model_c = ECAPA_TDNN(C=1024)
    # model_f = ECAPA_TDNN(C=1024)
    # ecapa_tdnn = CrossStitchNetwork(model_c, model_f)
    # ecapa_tdnn.load_state_dict(checkpoint['model'], False)

    # validation
    eer, cost = validate(ecapa_tdnn, val_dataloader, "/home/wldong/ffsvc/dev_test/trials_dev_keys")
    print(eer)
    with open('/home/zhy/FFSVC/validation.txt', 'a') as logs:
        logs.write("Mode_v1_b4, EER %.4f" %eer*100) 
        logs.write('/n')



if __name__ == '__main__':
    main()