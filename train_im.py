import os, random
import argparse
import numpy as np
import torch
import torch.optim as optim
import timm.models as tm_models
import pretrainedmodels as pt_models
import torch.nn.functional as F
from torch.cuda.amp import GradScaler,autocast
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import models
from dataloader import get_dataloader
from utils.visualizer import VisdomPlotter
from utils.loss import *
from utils.imb import ImbalancedDatasetSampler

vp = VisdomPlotter('8097', env='xf_im')

def train(args, model, train_loader, optimizer, epoch, scaler, criterion, scheduler=None):
    model.train()
    # criterion = torch.nn.CrossEntropyLoss() # FocalLoss().cuda() 
    bar = tqdm(train_loader, ncols=100)
    for data, target in bar:
        data, target = data.to(args.gpu), target.to(args.gpu)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output, target)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler != None:
            scheduler.step()
        bar.set_description('Train: [{}], Loss: {:.6f}'.format(epoch, loss.item()))
    return loss.item()


def val(args, model, test_loader, epoch=0):
    model.eval()
    test_loss = 0
    correct = 0
    bar = tqdm(test_loader, desc='Valid', ncols=100)
    with torch.no_grad():
        for data, target in bar:
            data, target = data.to(args.gpu), target.to(args.gpu)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_acc))

    return test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=137)
    parser.add_argument('--val_rate', type=float, default=0.2, help='val dataset rate')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer',
                        choices=['SGD', 'Adam', 'RMS'])
    parser.add_argument('--lr', type=float, default=0.016, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='efficientnet_b4')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA training')
    parser.add_argument('--seed', type=int, default=4567, metavar='S', help='random seed')
    parser.add_argument('--sufix', type=str, default='', help='sufix')
    parser.add_argument('--ckpt', type=str, default='saved/', help='pkl/pt/pth')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--im', type=str, default='DRW', help='long-tail balacing rule',
                        choices=['None', 'DRW', 'Reweight', 'Resample'])
    parser.add_argument('--loss', type=str, default='LDAM', help='loss func.',
                        choices=['LDAM', 'Focal', 'CE', 'CB'])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    multi_gpu = len(args.gpu.split(',')) > 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    train_loader, val_loader, cls_num_list = get_dataloader(args)
    scaler = GradScaler()
    
    if args.test_only:
        model = torch.load(args.ckpt, map_location=torch.device('cpu')).to(args.gpu)
        acc = val(args, model, val_loader)
        print('Val Acc = %.6f' % acc)
        return 

    model = tm_models.create_model(args.model, num_classes=args.num_classes, pretrained=True)
    # model = models.se_resnext50_32x4d(num_classes=137)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(args.gpu)


    if args.im == 'None':
        train_sampler = None  
        per_cls_weights = None 
    elif args.im == 'Resample':
        # train_sampler = ImbalancedDatasetSampler(train_dataset)
        per_cls_weights = None
    elif args.im == 'Reweight':
        train_sampler = None
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.gpu)
    elif args.im == 'DRW':
        train_sampler = None
        idx = args.epochs // 90
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.gpu)
    else:
        warnings.warn('Sample rule is not listed')
        return 
    
    print(cls_num_list)
    print(per_cls_weights)

    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda()
    elif args.loss == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=1, weight=per_cls_weights).cuda()
    elif args.loss == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1.).cuda()
    else:
        criterion = CBLoss(cls_num_list).cuda()
        # warnings.warn('Loss type is not listed')
        # return

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, eps=0.001)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(30*len(train_loader)), 0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], 0.2)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        #print("Lr = %.6f"%(optimizer.param_groups[0]['lr']))
        loss = train(args, model, train_loader, optimizer, epoch, scaler, criterion, scheduler)
        acc = val(args, model, val_loader, epoch)
        vp.add_scalar('Train Loss - %d'%args.seed, epoch, loss)
        vp.add_scalar('Val Acc - %d'%args.seed, epoch, acc)
        if acc > best_acc:
            best_acc = acc
            print('Saving a checkpoint...')
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module, "saved/%s%s.pt"%(args.model, args.sufix))
            else:
                torch.save(model, "saved/%s%s.pt"%(args.model, args.sufix))
    print("Best Acc = %.6f" % best_acc)

if __name__ == '__main__':
    main()