import torch
import torch.nn.functional as F
from sklearn.metrics import *
from scipy import interpolate

def ece_score(y_true,y_pred,bins=10):
    ece = 0.
    for i in range(bins):
        c_start,c_end = i/bins,(i+1)/bins
        mask = (c_start<=y_pred)&(y_pred<c_end)
        ni = mask.count_nonzero().item()
        if ni==0:
            continue
        acc,conf = y_true[mask].sum()/ni,y_pred[mask].mean()
        ece += ni*(acc-conf).abs()
    return ece.item()/len(y_pred)

def ACC(output,output_OOD,target,u=None,uo=None,region_len=100/3):
    with torch.no_grad():
        pred = torch.argmax(output,dim=1)
        correct = (pred==target)
        region_correct = (pred/region_len).long()==(target/region_len).long()
    acc = correct.sum().item()/len(target)
    region_acc = region_correct.sum().item()/len(target)
    split_acc = [0,0,0]

    # count number of classes for each region
    num_class = int(3*region_len)
    region_idx = (torch.arange(num_class)/region_len).long()
    region_vol = [
        num_class-torch.count_nonzero(region_idx).item(),
        torch.where(region_idx==1,True,False).sum().item(),
        torch.where(region_idx==2,True,False).sum().item()
    ]
    region_vol = [int(v*len(target)/num_class) for v in region_vol]

    for i in range(len(target)):
        split_acc[region_idx[target[i].item()]] += correct[i].item()
    split_acc = [split_acc[i]/region_vol[i] for i in range(3)]

    print('Classification ACC:')
    print('\t all \t =',acc)
    print('\t region  =',region_acc)
    print('\t head \t =',split_acc[0])
    print('\t med \t =',split_acc[1])
    print('\t tail \t =',split_acc[2])

def AUC(output,output_OOD,target,u=None,uo=None,region_len=100/3):
    with torch.no_grad():
        pred = torch.argmax(output,dim=1)
        correct = (pred==target)
    if u is None:
        p = F.softmax(output,dim=1)
        c = p.max(dim=1).values
    else:
        u = u/u.max()
        c = 1-u
    auc = roc_auc_score(correct.cpu(),c.cpu())
    split_auc = [0,0,0]
    for i in range(len(split_auc)):
        mask = ((target/region_len).long()==i)
        mask_correct = correct[mask]
        non_zero = mask_correct.count_nonzero().item()
        if non_zero==0 or non_zero==len(mask_correct):
            split_auc[i] = 0
        else:
            split_auc[i] = roc_auc_score(mask_correct.cpu(),c[mask].cpu())

    print('Failure Prediction AUC:')
    print('\t all \t =',auc)
    print('\t head \t =',split_auc[0])
    print('\t med \t =',split_auc[1])
    print('\t tail \t =',split_auc[2])

def FPR_95(output,output_OOD,target,u=None,uo=None,region_len=100/3):
    with torch.no_grad():
        pred = torch.argmax(output,dim=1)
        correct = (pred==target)
    if u is None:
        p = F.softmax(output,dim=1)
        c = p.max(dim=1).values
    else:
        u = u/u.max()
        c = 1-u
    fpr,tpr,_ = roc_curve(correct.cpu(),c.cpu())
    fpr_95 = float(interpolate.interp1d(tpr,fpr)(0.95))
    split_fpr_95 = [0,0,0]
    for i in range(len(split_fpr_95)):
        mask = ((target/region_len).long()==i)
        fpr,tpr,_ = roc_curve(correct[mask].cpu(),c[mask].cpu())
        split_fpr_95[i] = float(interpolate.interp1d(tpr,fpr)(0.95))

    print('Failure Prediction FPR-95:')
    print('\t all \t =',fpr_95)
    print('\t head \t =',split_fpr_95[0])
    print('\t med \t =',split_fpr_95[1])
    print('\t tail \t =',split_fpr_95[2])

def ECE(output,output_OOD,target,u=None,uo=None,region_len=100/3):
    with torch.no_grad():
        pred = torch.argmax(output,dim=1)
        correct = (pred==target)
    if u is None:
        p = F.softmax(output,dim=1)
        c = p.max(dim=1).values
    else:
        u = u/u.max()
        c = 1-u
    ece = ece_score(correct,c)
    split_ece = [0,0,0]
    for i in range(len(split_ece)):
        mask = ((target/region_len).long()==i)
        split_ece[i] = ece_score(correct[mask],c[mask])

    print('Failure Prediction ECE:')
    print('\t all \t =',ece)
    print('\t head \t =',split_ece[0])
    print('\t med \t =',split_ece[1])
    print('\t tail \t =',split_ece[2])

def TailOOD(output,output_OOD,target,u=None,uo=None,region_len=100/3):
    y,y_OOD = torch.ones(len(output)),torch.zeros(len(output_OOD)).long()
    if u is None:
        p = F.softmax(output,dim=1)
        c = p.max(dim=1).values
    else:
        u = u/u.max()
        c = 1-u
    tail_true = ((target/region_len).long()==2)
    tail = roc_auc_score(tail_true.cpu(),c.cpu())

    output = torch.cat([output,output_OOD],dim=0)
    y = torch.cat([y,y_OOD],dim=0)
    if u is None:
        p = F.softmax(output,dim=1)
        c = p.max(dim=1).values
    else:
        u = u/u.max()
        uo = uo/uo.max()
        u = torch.cat([u,uo],dim=0)
        c = 1-u
    ood = roc_auc_score(y.cpu(),c.cpu())

    print('Tail Detection AUC:',tail)
    print('OOD Detection AUC:',ood)
