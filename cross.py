import torch
import torch.nn as nn
import model

_alpha_keep = 0.9
_cross_stitch_unit = [[_alpha_keep, 1-_alpha_keep],
                      [1-_alpha_keep, _alpha_keep]]


class CrossStitchUnit(nn.Module):

    # size有两种类型，一种是中间的卷积层：batch_size * Channel * T，一种是全连接层：batch_size * n
    def __init__(self, size):# size is the input size
        super(CrossStitchUnit,self).__init__()
        assert len(size)==3 or len(size)==2
        if len(size) == 3:
            # 将十字绣单元的值转换成可以训练的parameter, shape -> Channel * 2 * 2
            self.cross_stitch_units = nn.Parameter(torch.Tensor([_cross_stitch_unit for i in range(size[1])]))
            self.cross_stitch_units.requires_grad_()
        elif len(size) == 2:
            # 将十字绣单元的值转换成可以训练的parameter, shape -> 1 * 2 * 2
            self.cross_stitch_units = nn.Parameter(torch.Tensor([_cross_stitch_unit for i in range(1)]))
            self.cross_stitch_units.requires_grad_()

    def forward(self, input1, input2):
        # input1 和 input2应当具有相同的形状
        assert input1.dim() == input2.dim() 
        output1, output2 = None, None
        # input1的维度为3，对应中间的卷积层， shape -> batch_size * Channel * T
        if input1.dim() == 3: 
            input_size = input1.size() # 记录input1的维度
            # 改变input1的维度，shape -> batch_size * Channel * 1 * T
            input1 = input1.view(input1.size(0), input1.size(1),1, -1) 
            # 改变input2的维度，shape -> batch_size * Channel * 1 * T
            input2 = input2.view(input1.size(0), input1.size(1),1, -1)
            # input_total.shape -> batch_size * Channel * 2 * T
            input_total = torch.cat((input1, input2), dim=2)
            # if self.cross_stitch_units is None:
            #     self.cross_stitch_units = torch.Tensor([_cross_stitch_unit for i in range(input1.size(1))])
            #     self.cross_stitch_units.requires_grad_()
            output_total = torch.matmul(self.cross_stitch_units, input_total)
            # 使用narrow分开output1和output2
            output1, output2 = torch.narrow(output_total, 2, 0, 1).view(input_size), \
                               torch.narrow(output_total, 2, 1, 1).view(input_size)
        elif input1.dim() == 2: #n*h, output after fc net
            input1 = input1.view(input1.size(0),  1, -1)
            input2 = input2.view(input1.size(0),  1, -1)
            input_total = torch.cat((input1, input2), dim=1)
            # if self.cross_stitch_units is None:
            #     self.cross_stitch_units = torch.Tensor(_cross_stitch_unit)
            #     self.cross_stitch_units.requires_grad_()
            output_total = torch.matmul(self.cross_stitch_units, input_total)
            output1, output2 = torch.narrow(output_total, 1, 0, 1).squeeze(), torch.narrow(output_total, 1, 1, 1).squeeze()
        return output1, output2


class CrossStitchNetwork(nn.Module):

    def __init__(self, source_architecture, target_architecture):
        super(CrossStitchNetwork,self).__init__()
        self.source_architecture = source_architecture
        self.target_architecture = target_architecture

        self.cross_stitch_units = nn.ModuleList()
        
        for m in self.source_architecture.children():
            if isinstance(m, (nn.Conv1d)):
                size = [None, 3, None]
                size[1] = m.out_channels
                self.cross_stitch_units.append(CrossStitchUnit(size))
                print(type(m), self.cross_stitch_units,'\n')
                print(sum(param.numel() for param in m.parameters())/1024/1024)
                break
            # elif isinstance(m, (nn.Linear)):
            #     size = [None, 1]
            #     self.cross_stitch_units.append(CrossStitchUnit(size))
            #     print(type(m), self.cross_stitch_units)
            # elif isinstance(m, (model.Bottle2neck)):
            #     size = [None, 3, None]
            #     size[1] = m.out_channels
            #     self.cross_stitch_units.append(CrossStitchUnit(size))
            #     print(type(m), self.cross_stitch_units)                    


    def forward(self, x):
        fbank_flag = True
        linear_flag = False
        B2N_flag = 1
        x1, x2 = x, x
        # consider about conv layer
        cross_unit_idx = 0
        for (m1, m2) in zip(self.source_architecture.children(), self.target_architecture.children()):
            # print(type(m1),type(m2))
            # print(x1.shape, x2.shape)
            if isinstance(m1,(nn.Sequential)):  
                if fbank_flag == True:
                    with torch.no_grad():
                        x1, x2 = m1(x)+1e-6, m1(x)+1e-6
                        x1, x2 = x1.log(), x2.log()
                        x1, x2 = (x1 - torch.mean(x1, dim=-1, keepdim=True)), (x2 - torch.mean(x2, dim=-1, keepdim=True))
                    fbank_flag = False
                    #print('1.torchfbank',x1.shape)
                if linear_flag == True:
                    t = x1.size()[-1]
                    global_x1, global_x2 = torch.cat((x1,torch.mean(x1,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x1,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1), \
                        torch.cat((x2,torch.mean(x2,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x2,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
                    w1, w2 = m1(global_x1), m2(global_x2)
                    mu1, mu2 = torch.sum(x1 * w1, dim=2), torch.sum(x2 * w2, dim=2)
                    sg1, sg2 = torch.sqrt( ( torch.sum((x1**2) * w1, dim=2) - mu1**2 ).clamp(min=1e-4) ), torch.sqrt( ( torch.sum((x2**2) * w2, dim=2) - mu2**2 ).clamp(min=1e-4) )
                    x1, x2 = torch.cat((mu1,sg1),1), torch.cat((mu2,sg2),1)
                    #print('ASP',x1.shape)

            
            if isinstance(m1, (model.FbankAug)):
                x1, x2 = m1(x1), m2(x2)
                #print('2.specaug',x1.shape)
            
            if isinstance(m1,(nn.Conv1d)):
                if B2N_flag == 4:
                    x1, x2 = m1(torch.cat((x1,o3,o5),dim=1)), m2(torch.cat((x2,o4,o6),dim=1))
                    # x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                    # cross_unit_idx += 1
                    linear_flag = True
                    #print('layer4',x1.shape)
                else:
                    x1, x2 = m1(x1), m2(x2)
                    x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                    # cross_unit_idx += 1
                    #print('3.conv1',x1.shape)

            if isinstance(m1,(nn.ReLU, nn.BatchNorm1d)):
                x1, x2 = m1(x1), m2(x2)
                #print('ReLU or BN',x1.shape)
            
            if isinstance(m1,(model.Bottle2neck)):
                if B2N_flag == 1:
                    o1, o2 = x1, x2 
                    x1, x2 = m1(x1), m2(x2)
                    # x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                    # cross_unit_idx += 1
                    B2N_flag +=1
                    #print('layer1',x1.shape)
                if B2N_flag == 2:
                    o3, o4 = x1, x2
                    x1, x2 = m1(x1+o1), m2(x2+o2)
                    # x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                    # cross_unit_idx += 1
                    B2N_flag +=1
                    #print('layer2',x1.shape)
                if B2N_flag == 3:
                    o5, o6 = x1, x2
                    x1, x2 = m1(x1+o1+o3), m2(x2+o2+o4)
                    # x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                    # cross_unit_idx += 1
                    B2N_flag +=1
                    #print('layer3',x1.shape)

            if isinstance(m1, (nn.Linear)):
                x1, x2 = m1(x1), m2(x2)
                # x1, x2 = self.cross_stitch_units[cross_unit_idx](x1, x2)
                # cross_unit_idx += 1
                #print('fc6')
                        

        x1, x2 = x1.contiguous().view(x1.size(0),-1), x2.contiguous().view(x2.size(0),-1)

        return x1, x2
