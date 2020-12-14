import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

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
    print('Sanity Check')
    cudnn.benchmark = True

    rgb_train_transform, rgb_test_transform = get_transform(args, train_rgb_augmentation, rgb_input_mean, 
                                                            rgb_input_std, scale_size, crop_size)
    
    flow_train_transform, flow_test_transform = get_transform(args, train_flow_augmentation, flow_input_mean, 
                                                            flow_input_std, scale_size, crop_size)

    rgb_model_path = os.path.join(args.save_dir, 'rgb_model_best.pth.tar')
    flow_model_path = os.path.join(args.save_dir, 'flow_model_best.pth.tar')
    fusion_model_path = os.path.join(args.save_dir, 'fusion_model_best.pth.tar')
    if args.modality in ['RGB', 'Joint']:
        print(("=> loading checkpoint '{}'".format(rgb_model_path)))
        ckpt = torch.load(rgb_model_path)
        model_rgb.load_state_dict(ckpt)
    if args.modality in ['Flow', 'Joint']:
        print(("=> loading checkpoint '{}'".format(flow_model_path)))
        ckpt = torch.load(flow_model_path)
        model_flow.load_state_dict(ckpt)
    if args.modality == 'Joint' and args.fusion_mode == 'mlp':
        print(("=> loading checkpoint '{}'".format(fusion_model_path)))
        ckpt = torch.load(fusion_model_path)
        model_fusion.load_state_dict(ckpt)

    data_length = 1
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args, args.data_dir,
                   args.val_list,
                   new_length=data_length,
                   random_shift=False,
                   test_mode=True,
                   rgb_transform=rgb_test_transform,
                   flow_transform=flow_test_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    epoch = 0
    with torch.no_grad():
        val_acc, val_std = validate(val_loader, model_rgb, model_flow, model_fusion, otam_loss, 0)
    print('val_acc: %.3f, val_std: %.3f' % (val_acc, val_std))


def validate(val_loader, model_rgb, model_flow, model_fusion, criterion, logger=None):
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
            
            '''
            query_ys = torch.arange(0,args.train_n).unsqueeze(1).repeat((1, args.n_queries)).view(-1).cuda()
            _, score_rgb = criterion(out_support_rgb, out_query_rgb, query_ys)
            _, score_flow = criterion(out_support_flow, out_query_flow, query_ys)
            
            score = (score_rgb + score_flow) / 2

            pred = torch.argmin(score, -1).detach().cpu().numpy()
            query_ys = query_ys.detach().cpu().numpy()
            acc = np.mean(pred==query_ys)
            accs.append(acc)
            pbar.set_postfix({"Acc":'{0:.2f}'.format(acc)})
            continue
            '''
            

            if args.modality == 'Joint':
                out_support_joint = model_fusion(out_support_rgb, out_support_flow)
                out_query_joint = model_fusion(out_query_rgb, out_query_flow)

            query_ys = torch.arange(0,args.train_n).unsqueeze(1).repeat((1, args.n_queries)).view(-1).cuda()

            if args.modality == 'RGB':
                loss, score = criterion(out_support_rgb, out_query_rgb, query_ys)
            elif args.modality == 'Flow':
                yyloss, score = criterion(out_support_flow, out_query_flow, query_ys)
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




def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h


if __name__ == '__main__':
    torch.manual_seed(0)
    main()

