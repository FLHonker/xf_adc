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

vp = None

def train(args, model, train_loader, optimizer, epoch, scaler):
    model.train()
    bar = tqdm(train_loader, ncols=80)
    for data, target in bar:
        data, target = data.to(args.gpu), target.to(args.gpu)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = loss_kd_regularization(output, target, alpha=0.95, T=20)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        bar.set_description('Train: [{}], Loss: {:.6f}'.format(epoch, loss.item()))
    return loss.item()

def val(args, model, test_loader, epoch=0):
    model.eval()
    test_loss = 0
    correct = 0
    bar = tqdm(test_loader, desc='Valid', ncols=80)
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
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='efficientnet_b4')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA training')
    parser.add_argument('--seed', type=int, default=6786, metavar='S', help='random seed')
    parser.add_argument('--step_sz', type=int, default=30)
    parser.add_argument('--sufix', type=str, default='', help='sufix')
    parser.add_argument('--ckpt', type=str, default='saved/', help='pkl/pt/pth')
    parser.add_argument('--test_only', action='store_true', default=False)

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

    vp = VisdomPlotter('8097', env='xf_adc')

    print(args)

    train_loader, val_loader, _ = get_dataloader(args)
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
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_sz, 0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], 0.1)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        #print("Lr = %.6f"%(optimizer.param_groups[0]['lr']))
        loss = train(args, model, train_loader, optimizer, epoch, scaler)
        scheduler.step()
        acc = val(args, model, val_loader, epoch)
        vp.add_scalar('Loss-%s'%args.model, epoch, loss)
        vp.add_scalar('Acc-%s'%args.model, epoch, acc)
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