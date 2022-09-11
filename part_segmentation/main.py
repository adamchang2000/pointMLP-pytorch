from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.data_util import YCBDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
import cv2

from time import perf_counter

classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(args, io):

    # ============= Model ===================
    num_part = 22
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    '''Resume or not'''
    if args.resume:
        state_dict = torch.load("checkpoints/%s/best_acc_model.pth" % args.exp_name,
                                map_location=torch.device('cpu'))['model']
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print("Resume training model...")
        print(torch.load("checkpoints/%s/best_acc_model.pth" % args.exp_name).keys())
    else:
        print("Training from scratch...")

    # =========== Dataloader =================
    train_data = YCBDataset(npoints=args.num_points, split='train', normalize=False)
    print("The number of training data is:%d", len(train_data))

    test_data = YCBDataset(npoints=args.num_points, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    print("Using {0} workers!".format(args.workers))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # ============= Optimizer ================
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'cos':
        print("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    else:
        print("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # ============= Training =================
    best_acc = 0
    #best_class_iou = 0
    #best_instance_iou = 0
    num_part = 22
    num_classes = 16

    for epoch in range(args.epochs):

        train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io)

        test_metrics = test_epoch(test_loader, model, epoch, num_part, num_classes, io)

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/%s/best_acc_model.pth' % args.exp_name)

        # 2. when get the best instance_iou, save the model:
        # if test_metrics['shape_avg_iou'] > best_instance_iou:
        #     best_instance_iou = test_metrics['shape_avg_iou']
        #     io.cprint('Max instance iou:%.5f' % best_instance_iou)
        #     state = {
        #         'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        #         'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou}
        #     torch.save(state, 'checkpoints/%s/best_insiou_model.pth' % args.exp_name)

        # 3. when get the best class_iou, save the model:
        # first we need to calculate the average per-class iou
        # class_iou = 0
        # for cat_idx in range(16):
        #     class_iou += total_per_cat_iou[cat_idx]
        # avg_class_iou = class_iou / 16
        # if avg_class_iou > best_class_iou:
        #     best_class_iou = avg_class_iou
        #     # print the iou of each class:
        #     for cat_idx in range(16):
        #         io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
        #     io.cprint('Max class iou:%.5f' % best_class_iou)
        #     state = {
        #         'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        #         'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou}
        #     torch.save(state, 'checkpoints/%s/best_clsiou_model.pth' % args.exp_name)

    # report best acc, ins_iou, cls_iou
    io.cprint('Final Max Acc:%.5f' % best_acc)
    #io.cprint('Final Max instance iou:%.5f' % best_instance_iou)
    #io.cprint('Final Max class iou:%.5f' % best_class_iou)
    # save last model
    state = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_acc': best_acc}
    torch.save(state, 'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))


def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = defaultdict(lambda: list())
    model.train()

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        # t1 = perf_counter()

        # target: b,n
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50

        # t2 = perf_counter()

        # print("Forward pass", t2 - t1)

        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])

        # t2_5 = perf_counter()

        # print("Loss Function", t2_5 - t2)

        # instance iou without considering the class average at each batch_size:
        # batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        # batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # t3 = perf_counter()

        # print("IOU stuff", t3 - t2_5)

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # t4 = perf_counter()

        # print("Optimizer step and backwards", t4 - t3)

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]   # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum
        # shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration

        # Note: We do not need to calculate per_class iou during training

        # t5 = perf_counter()

        # print("Other stuff", t5 - t4)

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5
    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch+1, train_loss * 1.0 / count,
                                                                       metrics['accuracy'], metrics['shape_avg_iou'])
    io.cprint(outstr)


def test_epoch(test_loader, model, epoch, num_part, num_classes, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    # shape_ious = 0.0
    # final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        # batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
            # final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
            final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

        # total iou of current batch in each process:
        # batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        loss = torch.mean(loss)
        # shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    # for cat_idx in range(16):
    #     if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
    #         final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    #metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Test %d, loss: %f, test acc: %f' % (epoch + 1, test_loss * 1.0 / count,
                                                                    metrics['accuracy'])

    io.cprint(outstr)

    return metrics

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts

colors = [(255, 255, 255), (51, 102, 255), (153, 51, 255), (204, 0, 204), (255, 204, 0), (153, 204, 0)
            , (0, 102, 102), (51, 102, 0), (153, 0, 204), (102, 0, 51), (102, 255, 255), (102, 255, 153)
            , (153, 51, 0), (102, 153, 153), (102, 51, 0), (153, 153, 102), (255, 204, 153), (255, 102, 102), (0, 255, 153)
            , (102, 0, 102), (153, 255, 51), (51, 102, 153)]

vis_dir = "visualization"

if not os.path.isdir(vis_dir):
    os.mkdir(vis_dir)

def test(args, io):
    # Dataloader
    test_data = YCBDataset(npoints=args.num_points, split='test', normalize=False, return_project_data=True, crop_object=False)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # Try to load models
    num_part = 22
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 22
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    #shape_ious = []

    for batch_id, (points, label, target, norm_plt, project_data) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)

        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        intr = project_data["intr"]
        intr = [x.cpu().detach().numpy() for x in intr]
        cam_cx, cam_cy, cam_fx, cam_fy = intr

        cloud_mean = project_data["cloud_mean"].cpu().detach().numpy()

        color_img = project_data["img"].squeeze(0).cpu().detach().numpy()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        color_img_2 = np.copy(color_img)

        points = points.cpu().detach().numpy()
        points = points.squeeze(0).transpose((1, 0))
        cloud_mean = cloud_mean.squeeze()

        points = points + cloud_mean

        projected_pts = project_points(points, cam_fx, cam_fy, cam_cx, cam_cy)

        preds = torch.argmax(seg_pred, 2).squeeze(0).cpu().detach().numpy()

        gts = target.squeeze().cpu().detach().numpy()

        for (x, y), c in zip(projected_pts, preds):
            r, g, b = colors[c]
            color_img = cv2.circle(color_img, (int(x), int(y)), radius=1, color=(b,g,r), thickness=-1)

        for (x, y), c in zip(projected_pts, gts):
            r, g, b = colors[c]
            color_img_2 = cv2.circle(color_img_2, (int(x), int(y)), radius=1, color=(b,g,r), thickness=-1)

        output_filename = '{0}/{1}.png'.format(vis_dir, batch_id)
        cv2.imwrite(output_filename, color_img)

        output_filename = '{0}/{1}_GT.png'.format(vis_dir, batch_id)
        cv2.imwrite(output_filename, color_img_2)

        # # instance iou without considering the class average at each batch_size:
        # batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # shape_ious += batch_shapeious  # iou +=, equals to .append

        # # per category iou at each batch_size:
        # for shape_idx in range(seg_pred.size(0)):  # sample_idx
        #     cur_gt_label = label[shape_idx]  # label[sample_idx]
        #     total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
        #     total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    #metrics['shape_avg_iou'] = np.mean(shape_ious)
    # for cat_idx in range(16):
    #     if total_per_cat_seen[cat_idx] > 0:
    #         total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    # class_iou = 0
    # for cat_idx in range(16):
    #     class_iou += total_per_cat_iou[cat_idx]
    #     io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    # avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f' % (metrics['accuracy'])
    io.cprint(outstr)

def saliency_vis(args, io):

    saliency_dir = "saliency"
    if not os.path.isdir(saliency_dir):
        os.mkdir(saliency_dir)

    # Dataloader
    test_data = YCBDataset(npoints=12800, split='test', normalize=False, return_project_data=True, crop_object=False)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # Try to load models
    num_part = 22
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 22
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    #shape_ious = []

    for batch_id, (points, label, target, norm_plt, project_data) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)

        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        points = Variable(points, requires_grad=True)
        norm_plt = Variable(norm_plt, requires_grad=True)

        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        intr = project_data["intr"]
        intr = [x.cpu().detach().numpy() for x in intr]
        cam_cx, cam_cy, cam_fx, cam_fy = intr

        cloud_mean = project_data["cloud_mean"].cpu().detach().numpy()

        color_img = project_data["img"].squeeze(0).cpu().detach().numpy()
        h, w, _ = color_img.shape

        cloud_mean = cloud_mean.squeeze()

        points_centered = points.cpu().detach().numpy().squeeze(0).transpose((1, 0)) + cloud_mean

        projected_pts = project_points(points_centered, cam_fx, cam_fy, cam_cx, cam_cy)

        preds = torch.argmax(seg_pred, 2).squeeze(0)

        for i in range(21):
            mask = preds == (i + 1)

            if torch.count_nonzero(mask) < 30:
                continue

            pred_logits = seg_pred[0]

            pred_logits = pred_logits[mask][:, i + 1]

            dcloud, dnormals = torch.autograd.grad(pred_logits, (points, norm_plt), torch.ones(pred_logits.shape).cuda(), retain_graph=True)

            dcloud = dcloud.squeeze(0).transpose(0, 1)
            dnormals = dnormals.squeeze(0).transpose(0, 1)

            dcloud, _ = torch.max(torch.abs(dcloud), dim=1)
            dnormals, _ = torch.max(torch.abs(dnormals), dim=1)

            dcloud = dcloud.cpu().detach().numpy()
            dnormals = dnormals.cpu().detach().numpy()

            dcloud = dcloud / np.max(dcloud)
            dnormals = dnormals / np.max(dnormals)

            doutcloud = np.zeros((h, w)).astype(np.uint16)
            doutnormals = np.zeros((h, w)).astype(np.uint16)

            for (x, y), c in zip(projected_pts, dcloud):
                doutcloud = cv2.circle(doutcloud, (int(x), int(y)), radius=1, color=int(c * 65535), thickness=-1)

            for (x, y), c in zip(projected_pts, dnormals):
                doutnormals = cv2.circle(doutnormals, (int(x), int(y)), radius=1, color=int(c * 65535), thickness=-1)

            output_filename = '{0}/{1}_{2}_cloud.png'.format(saliency_dir, batch_id, i + 1)
            cv2.imwrite(output_filename, doutcloud)

            output_filename = '{0}/{1}_{2}_normals.png'.format(saliency_dir, batch_id, i + 1)
            cv2.imwrite(output_filename, doutnormals)


        # # instance iou without considering the class average at each batch_size:
        # batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # shape_ious += batch_shapeious  # iou +=, equals to .append

        # # per category iou at each batch_size:
        # for shape_idx in range(seg_pred.size(0)):  # sample_idx
        #     cur_gt_label = label[shape_idx]  # label[sample_idx]
        #     total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
        #     total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    #metrics['shape_avg_iou'] = np.mean(shape_ious)
    # for cat_idx in range(16):
    #     if total_per_cat_seen[cat_idx] > 0:
    #         total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    # class_iou = 0
    # for cat_idx in range(16):
    #     class_iou += total_per_cat_iou[cat_idx]
    #     io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    # avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f' % (metrics['accuracy'])
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='pointMLP')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval',default=False, action="store_true", help='evaluate the model')
    parser.add_argument('--saliency', default=False, action="store_true", help='visualize saliency')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', default=False, action="store_true",
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='acc',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name

    _init_()

    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if args.saliency:
        saliency_vis(args, io)
    elif args.eval:
        test(args, io)
    else:
        train(args, io)