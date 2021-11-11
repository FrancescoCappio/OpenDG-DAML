import sys
import argparse


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.append('.')

import dalib.vision.datasets as datasets
from tools.utils import AverageMeter, accuracy

import os
import numpy as np

from tools.daml_utils import resnet18_fast, ClassifierFast


def main(args):
    """
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    """
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_tranform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if args.data == 'PACS':
        n1 = 0
        n2 = 1
        n3 = 1
        n4 = 0
        n5 = 1
        n6 = 1
        n7 = 1
    elif args.data == 'OfficeHome':
        n1 = 3
        n2 = 6
        n3 = 11
        n4 = 1
        n5 = 2
        n6 = 3
        n7 = 11

    S123 = [i for i in range(n1)]
    S12 = [i + n1 for i in range(n2)]
    S13 = [i + n1 + n2 for i in range(n2)]
    S23 = [i + n1 + n2 * 2 for i in range(n2)]
    S1 = [i + n1 + n2 * 3 for i in range(n3)]
    S2 = [i + n1 + n2 * 3 + n3 for i in range(n3)]
    S3 = [i + n1 + n2 * 3 + n3 * 2 for i in range(n3)]

    ST1 = [S123[i] for i in range(n4)] \
        + [S12[i] for i in range(n5)] + [S13[i] for i in range(n5)] + [S23[i] for i in range(n5)] \
        + [S1[i] for i in range(n6)] + [S2[i] for i in range(n6)] + [S3[i] for i in range(n6)]

    TT = [i + n1 + n2 * 3 + n3 * 3 for i in range(n7)]

    source_classes = [[], [], []]
    source_classes[0] = S1 + S12 + S13 + S123
    source_classes[1] = S2 + S12 + S23 + S123
    source_classes[2] = S3 + S13 + S23 + S123

    all_target_classes = ST1 + TT
    print(all_target_classes)
    
    dataset = datasets.__dict__[args.data]

    num_classes = n1 + n2 * 3 + n3 * 3

    backbone1 = resnet18_fast()
    backbone2 = resnet18_fast()
    backbone3 = resnet18_fast()

    classifier = ClassifierFast(backbone1, backbone2, backbone3, num_classes).cuda()
    for weight in classifier.parameters():
        weight.fast = None

    pretrained_dir = 'runs/daml-' + args.source + '-' + args.target + '_best_val.tar'

    classifier.load_state_dict(torch.load(pretrained_dir))
    the_target = args.target
    print('Target Domain: ', the_target)

    raw_output_dict = {}
    target_dict = {}
    targets = []
    all_paths = []

    # for each class in target classes extract predictions from source models for target samples of that class
    for class_index in all_target_classes:
        the_class = [class_index]

        the_target_dataset = dataset(root=args.root, task=the_target, filter_class=the_class, split='all',
                                    transform=val_tranform)
        print(f"Class {class_index}, size: {len(the_target_dataset)}")
        the_target_loader = DataLoader(the_target_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers, drop_last=False)
        the_target_list = [the_target_loader]

        output, target, class_paths = get_raw_output(the_target_list, classifier, num_classes)
        all_paths.extend(class_paths)
        raw_output_dict[class_index] = output
        target_dict[class_index] = target
        targets.append(target)

    import ipdb; ipdb.set_trace() 
    targets = torch.cat(targets)
    outlier_targets = (targets == num_classes).float()

    T = 1.0
    tsm_output_dict = {}
    outlier_indi_dict = {}
    outlier_indis = []

    # for each target class, perform ensemble of source models predictions with a non weighted average, 
    # also extract the maximum confidence as "indicator" for each sample

    correct = 0
    total = 0
    idx = 0
    predictions = {}
    for class_index in all_target_classes:
        raw_output = raw_output_dict[class_index]

        output, indicator = get_new_output(raw_output, T)

        predicted_val, predicted_cls = output.max(dim=1)
        if class_index not in TT:
            correct += (predicted_cls==class_index).sum()
            total += len(output)
        
        for p_cls, p_conf in zip(predicted_cls, predicted_val):
            predictions[idx] = {'cls': p_cls, "conf": p_conf}
            idx += 1

        tsm_output_dict[class_index] = output
        outlier_indi_dict[class_index] = indicator
        outlier_indis.append(indicator)

    print(f"Closed set acc: {(correct/total)*100}")
    with open("predictions.txt", "w") as out_f:
        for k in predictions.keys():
            out_f.write(f"{k},{predictions[k]['cls']},{predictions[k]['conf']}\n")

    outlier_indis = torch.cat(outlier_indis)

    # get range of confidence values, divide it in 10 in order to obtain 10 possible thresholds values
    thd_min = torch.min(outlier_indis)
    thd_max = torch.max(outlier_indis)
    outlier_range = [thd_min + (thd_max - thd_min) * i / 9 for i in range(10)]

    best_overall_acc = 0.0
    best_thred_acc = 0.0
    best_overall_Hscore = 0.0
    best_thred_Hscore = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_overall_caa = 0.0
    best_thred_caa = 0.0

    # let's compute auroc 
    auroc = get_auroc(scores_id=outlier_indis[targets<num_classes].cpu().numpy(),scores_ood=outlier_indis[targets==num_classes].cpu().numpy())
    print("Auroc: ", auroc)

    # for each possible threshold value
    for outlier_thred in outlier_range:
        acc_dict = {}

        # for each target class, get sources ensembled predictions for that class and max confidences for that class 
        # get accuracy in the prediction of that class
        for class_index in all_target_classes:
            tsm_output = tsm_output_dict[class_index]
            outlier_indi = outlier_indi_dict[class_index]
            target = target_dict[class_index]
            acc = get_acc(tsm_output, outlier_indi, outlier_thred, target)
            acc_dict[class_index] = acc

        # overall accuracy is not simply the average class accuracy. On the contrary we take into account also class cardinalities
        # and we obtain accuracy over the whole dataset 
        overall_acc = (np.sum([acc.sum.item() for acc in acc_dict.values()]) / np.sum([acc.count for acc in acc_dict.values()])).item()

        # insider acc is overall accuracy over known classes (the same as overall_acc, but without considering outlier classes)
        insider = (np.sum([acc_dict[Cl].sum.item() for Cl in ST1]) / np.sum([acc_dict[Cl].count for Cl in ST1])).item()
        # outsider acc is overall accuracy for outlier classes 
        outsider = (np.sum([acc_dict[Cl].sum.item() for Cl in TT]) / np.sum([acc_dict[Cl].count for Cl in TT])).item()
        # compute Hscore
        overall_Hscore = 2.0 * insider * outsider / (insider + outsider)

        # overall caa is the averace of class accuracies
        overall_caa = np.mean([acc.avg.item() for acc in acc_dict.values()])

        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            best_thred_acc = outlier_thred
        if overall_Hscore > best_overall_Hscore:
            best_overall_Hscore = overall_Hscore
            best_thred_Hscore = outlier_thred
            best_known_acc = insider
            best_unknown_acc = outsider
        if overall_caa > best_overall_caa:
            best_overall_caa = overall_caa
            best_thred_caa = outlier_thred


    print('Best Overall Acc: %.2f' % (best_overall_acc), 'Best Acc threshold: %.3f' % (best_thred_acc),
    '\nBest Overall Hscore: %.2f' % (best_overall_Hscore), 'Best Hscore threshold: %.3f' % (best_thred_Hscore),
    '\nKnown acc: %.2f ' % (best_known_acc), 'Unk acc: %.2f' % (best_unknown_acc),
    '\nBest Overall Caa: %.2f' % (best_overall_caa), 'Best Caa threshold: %.3f' % (best_thred_caa))
    


def get_raw_output(val_loader, model, num_classes):
    model.eval()
    output_sum = []
    target_sum = []
    all_paths = []

    with torch.no_grad():
        for the_loader in val_loader:
            for i, (images, target, _, paths) in enumerate(the_loader):
                images = images.cuda()
                target = target.cuda()
                all_paths.extend(list(paths))
                outlier_flag = (target > (num_classes - 1)).float()
                target = target * (1 - outlier_flag) + num_classes * outlier_flag
                target = target.long()

                output, _ = model(images)
                output_sum.append(output)

                target_sum.append(target)

    output_sum = [torch.cat([output_sum[j][i] for j in range(len(output_sum))], dim=0) for i in range(3)]

    target_sum = torch.cat(target_sum)
    return output_sum, target_sum, all_paths


def get_new_output(raw_output, T):

    output = [F.softmax(headout/T, dim=1) for headout in raw_output]
    output = torch.mean(torch.stack(output), 0)
    max_prob, max_index = torch.max(output, 1)
    return output, max_prob



def get_acc(tsm_output, outlier_indi, outlier_thred, target):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # we use the outlier_indicator to compute a score for the prediction of the outlier class
    # the score is simply 0 if the outlier class is not predicted (max confidence is higher than threshold),
    # otherwise it takes value 1
    outlier_pred = (outlier_indi < outlier_thred).float()
    outlier_pred = outlier_pred.view(-1, 1)
    # we concatenate prediction of outlier class with predictions of other classes
    output = torch.cat((tsm_output, outlier_pred.cuda()), dim=1)
    # measure accuracy and record loss
    # we compute accuracy for current class 
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    top1.update(acc1[0], output.shape[0])
    return top1


def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)

if __name__ == '__main__':

    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Open Domain Generalization')
    parser.add_argument('--root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', type=str, help='source domain(s)')
    parser.add_argument('-t', '--target', type=str, help='target domain(s)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id ')

    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

