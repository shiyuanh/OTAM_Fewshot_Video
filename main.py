import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import sklearn
import scipy.stats
from scipy.stats import t
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F

from dataset import TSNDataSet
from models import OTAM, Fusion
from transforms import *
from opts import parser
from loss import OTAM_Loss, Baseline_Loss
from tqdm import tqdm

best_prec1 = 0

def get_model(args, num_class, modality):
    model = OTAM(num_class, args.num_segments, modality,
                base_model=args.arch, new_length=1,
                dropout=args.dropout, partial_bn=not args.no_partialbn)
    input_mean = model.input_mean
    input_std = model.input_std
    #policies = model_rgb.get_optim_policies()
    train_augmentation = model.get_augmentation()
    model = torch.nn.DataParallel(model).cuda()
    return model, train_augmentation, input_mean, input_std

def get_transform(args, train_augmentation, input_mean, input_std, scale_size=256, crop_size=224):
    train_transform=torchvision.transforms.Compose([
                   train_augmentation,
                   Stack(roll=args.arch == 'BNInception'),
                   ToTorchFormatTensor(div=args.arch != 'BNInception'),
                   GroupNormalize(input_mean, input_std)])
    test_transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(input_mean, input_std)])
    return train_transform, test_transform

def get_optimizer(args, model):
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'base_model' in key:
            decay_mult = 0.0 if 'bias' in key else 1.0
            lr_mult = 1.0 # for cls, just finetune. if '.fc.' in key: lr_mult = 1.0
            params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    optimizer = torch.optim.SGD(
        params,
        weight_decay=args.weight_decay,
        momentum=args.momentum)
    return optimizer


def main():
    global args, best_prec1
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    args.save_dir = os.path.join(args.save_dir, '_'.join([args.modality, str(args.num_segments), args.fusion_mode]))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    

    if args.dataset == 'something':
        num_class = 5
    elif args.dataset == 'hmdb51':
        num_class = 5
    elif args.dataset == 'kinetics':
        num_class = 5
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model_rgb, train_rgb_augmentation, rgb_input_mean, rgb_input_std = get_model(args, num_class, 'RGB')  
    model_flow, train_flow_augmentation, flow_input_mean, flow_input_std = get_model(args, num_class, 'Flow')
    model_fusion = Fusion(mode=args.fusion_mode).cuda()

    if args.loss_type == 'otam':
        otam_loss = OTAM_Loss(smooth_param=args.smooth_param, sim_metric='cosine').cuda()
    elif args.loss_type == 'baseline':
        otam_loss = Baseline_Loss()



    crop_size = 224
    scale_size = 256

    cudnn.benchmark = True

    rgb_train_transform, rgb_test_transform = get_transform(args, train_rgb_augmentation, rgb_input_mean, 
                                                            rgb_input_std, scale_size, crop_size)
    
    flow_train_transform, flow_test_transform = get_transform(args, train_flow_augmentation, flow_input_mean, 
                                                            flow_input_std, scale_size, crop_size)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    
    data_length = 1

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args,'/data/users/shiyuan/HMDB51/coviar_input/TV_L1', 
                   args.train_list,
                   new_length=data_length,
                   fix_seed=False,
                   rgb_transform=rgb_train_transform,
                   flow_transform=flow_train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args,'/data/users/shiyuan/HMDB51/coviar_input/TV_L1',
                   args.val_list,
                   new_length=data_length,
                   random_shift=False,
                   test_mode=True,
                   rgb_transform=rgb_test_transform,
                   flow_transform=flow_test_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    optim_rgb = get_optimizer(args, model_rgb)
    optim_flow = get_optimizer(args, model_flow)
    optim_fusion = None
    if args.fusion_mode == 'mlp':
        params_dict = dict(model_fusion.named_parameters())
        params = []
        for key, value in params_dict.items():
            decay_mult = 0.0 if 'bias' in key else 1.0
            lr_mult = 1.0 # for cls, just finetune. if '.fc.' in key: lr_mult = 1.0
            params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]
        optim_fusion = torch.optim.SGD(params, weight_decay=args.weight_decay, momentum=args.momentum)

    if args.evaluate:
        epoch = 0
        with torch.no_grad():
            val_acc, val_std = validate(val_loader, model_rgb, model_flow, model_fusion, otam_loss, (epoch + 1) * len(train_loader))
        return

    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optim_rgb, epoch, args.lr_steps)
        lr = adjust_learning_rate(optim_flow, epoch, args.lr_steps)
        if args.fusion_mode == 'mlp':
            adjust_learning_rate(optim_fusion, epoch, args.lr_steps)

        # train for one epoch
        avg_acc = train(train_loader, model_rgb, model_flow, model_fusion, optim_rgb, optim_flow, optim_fusion, otam_loss, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                val_acc, val_std = validate(val_loader, model_rgb, model_flow, model_fusion, otam_loss, (epoch + 1) * len(train_loader))
            if best_prec1 < val_acc:
                best_epoch = epoch
                best_prec1 = val_acc
                if args.modality in ['RGB', 'Joint']:
                    save_checkpoint(model_rgb.state_dict(), is_best=True, modality='RGB', filename='checkpoint.pth.tar')
                if args.modality in ['Flow', 'Joint']:
                    save_checkpoint(model_flow.state_dict(), is_best=True, modality='Flow', filename='checkpoint.pth.tar')
                if args.modality == 'Joint' and args.fusion_mode == 'mlp':
                    save_checkpoint(model_fusion.state_dict(), is_best=True, modality='Fusion', filename='checkpoint.pth.tar')

            message = 'Epoch %d lr %f Val_Acc %.3f Best_Acc %.3f @ Epoch %d' % (epoch, lr, val_acc, best_prec1, best_epoch)
            print(message)

def train(train_loader, model_rgb, model_flow, model_fusion, optim_rgb, optim_flow, optim_fusion, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if args.no_partialbn:
        model_rgb.module.partialBN(False)
        model_flow.module.partialBN(False)
    else:
        model_flow.module.partialBN(True)

    # switch to train mode
    model_rgb.train()
    model_flow.train()
    model_fusion.train()

    end = time.time()
    acc = AverageMeter()
    losses = AverageMeter()
    with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
        for i, data in enumerate(pbar):
            data_time.update(time.time() - end)
            # compute output
            loss_tot = 0

            support_xs_rgb, support_xs_flow, support_ys, query_xs_rgb, query_xs_flow, query_ys = data
            
            if args.modality in ['RGB', 'Joint']:
                out_support_rgb, out_query_rgb = compute_feat(model_rgb, support_xs_rgb, query_xs_rgb)
            if args.modality in ['Flow', 'Joint']:
                out_support_flow, out_query_flow = compute_feat(model_flow, support_xs_flow, query_xs_flow)
            #print(out_support.shape, out_query.shape)


            if args.modality == 'Joint':
                out_support_joint = model_fusion(out_support_rgb, out_support_flow)
                out_query_joint = model_fusion(out_query_rgb, out_query_flow)

            query_ys = torch.arange(0,args.train_n).unsqueeze(1).repeat((1, args.n_queries)).view(-1).cuda()

            if args.modality == 'RGB':
                loss, score = criterion(out_support_rgb, out_query_rgb, query_ys)
            elif args.modality == 'Flow':
                loss, score = criterion(out_support_flow, out_query_flow, query_ys)
            elif args.modality == 'Joint':
                loss, score = criterion(out_support_joint, out_query_joint, query_ys)

            optim_rgb.zero_grad()
            optim_flow.zero_grad()
            if args.fusion_mode == 'mlp':
                optim_fusion.zero_grad()

            loss.backward()
            
            if args.modality in ['RGB', 'Joint']:
                optim_rgb.step()
            if args.modality in ['Flow', 'Joint']:
                optim_flow.step()
            if args.modality == 'Joint' and args.fusion_mode == 'mlp':
                optim_fusion.step()
            # measure elapsed time

            batch_time.update(time.time() - end)
            end = time.time()

            pred = torch.argmin(score, -1).detach().cpu().numpy()
            query_ys = query_ys.detach().cpu().numpy()


            acc.update(np.mean(pred==query_ys))
            losses.update(loss.item())

            pbar.set_postfix({"Train Acc":'{0:.3f}'.format(acc.avg),
                          "Train Loss" :'{0:.3f}'.format(losses.avg)})
    #avg_acc, _ = mean_confidence_interval(avg_acc)
    message = 'Epoch {} Train_Acc {acc.avg:.3f} Train_Loss {losses.avg:.3f}'.format(epoch, acc=acc, losses=losses)
    print(message)
    return acc.avg, losses.avg


def validate(val_loader, model_rgb, model_flow, model_fusion, criterion, logger=None):
    batch_time = AverageMeter()
    pred = []
    model_rgb.eval()
    model_flow.eval()
    model_fusion.eval()
    correct = 0
    tot = 0
    accs = []

    with tqdm(val_loader, total=len(val_loader), leave=False) as pbar:
        for i, data in enumerate(pbar):
            
            support_xs_rgb, support_xs_flow, support_ys, query_xs_rgb, query_xs_flow, query_ys = data

            if args.modality in ['RGB', 'Joint']:
                out_support_rgb, out_query_rgb = compute_feat(model_rgb, support_xs_rgb, query_xs_rgb)
            if args.modality in ['Flow', 'Joint']:
                out_support_flow, out_query_flow = compute_feat(model_flow, support_xs_flow, query_xs_flow)
            #print(out_support.shape, out_query.shape)

            if args.modality == 'Joint':
                out_support_joint = model_fusion(out_support_rgb, out_support_flow)
                out_query_joint = model_fusion(out_query_rgb, out_query_flow)

            query_ys = torch.arange(0,args.train_n).unsqueeze(1).repeat((1, args.n_queries)).view(-1).cuda()

            if args.modality == 'RGB':
                loss, score = criterion(out_support_rgb, out_query_rgb, query_ys)
            elif args.modality == 'Flow':
                loss, score = criterion(out_support_flow, out_query_flow, query_ys)
            elif args.modality == 'Joint':
                loss, score = criterion(out_support_joint, out_query_joint, query_ys)


            pred = torch.argmin(score, -1).detach().cpu().numpy()
            query_ys = query_ys.detach().cpu().numpy()
            acc = np.mean(pred==query_ys)

            accs.append(acc)
            pbar.set_postfix({"Acc":'{0:.2f}'.format(acc)})
                                            
    avg_acc, std = mean_confidence_interval(accs)
    return avg_acc, std 




def compute_feat(model, support, query):
    B, N, T, C, H, W = support.shape
    assert T == args.num_segments
    assert N == args.train_k * args.train_n
    M = query.shape[1]
    assert M == args.train_n * args.n_queries
    images = torch.cat([support, query], dim=1).view(-1, C, H, W)
    features = model(images)
    _, D = features.shape
    features = features.view(-1, T, D)
    out_support, out_query = torch.split(features, [B*N, B*M])
    out_support = out_support.view(B, N, T, D)
    out_query = out_query.view(B, M, T, D)
    return out_support, out_query


def save_checkpoint(state, is_best, modality, filename='checkpoint.pth.tar'):
    #filename = '_'.join((args.modality.lower(), filename))
    #filename = os.path.join(args.save_dir, filename)
    #torch.save(state, filename)
    if is_best:
        best_name = '_'.join((modality.lower(), 'model_best.pth.tar'))
        filename = os.path.join(args.save_dir, best_name)
        torch.save(state, filename)
        #shutil.copyfile(filename, best_name)


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']
    return lr


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
