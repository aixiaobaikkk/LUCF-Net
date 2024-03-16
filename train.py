import torch.backends.cudnn as cudnn
import argparse
import logging
import random
import sys
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision import transforms
from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.loss import DiceLoss,LovaszSoftmax,OhemCrossEntropy
from utils.utils import val_single_volume
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from networks.LUCF_Net import LUCF_Net

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,
                    default='LUCF_Net', help='')
parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz_new', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/test_vol_h5_new', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./data/synapse/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--in_chans', type=int,
                    default=1, help='input channel')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--output_dir', type=str,
                    default="model_pth/synapse/", help='output')
parser.add_argument('--is_pretrained', type=bool,
                    default=False, help='whether loading pretrained weights')
parser.add_argument('--pretrained_pth', type=str,
                    default=r'', help='pretrained model weights')
parser.add_argument('--n_gpu', type=int,
                    default=1, help='total gpu')
parser.add_argument('--gpu', type=str, default='0', help='gpu using')

parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--save_interval', type=int, default=50,
                    help='save model interval')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='')


args = parser.parse_args()

def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,
                              nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size],
                                     case=case_name, z_spacing=args.z_spacing,model_name=args.model_name)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    alpha = 0.2

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)
    model.train()

    Lovasz_loss = LovaszSoftmax()
    dice_loss = DiceLoss(num_classes)
    ohem_ce_loss = OhemCrossEntropy()
    # ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)#optional
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            p1, p2, p3, p4 = model(image_batch)
            outputs = p1 + p2 + p3 + p4

            loss_iou1 = Lovasz_loss(p1, label_batch)
            loss_iou2 = Lovasz_loss(p2, label_batch)
            loss_iou3 = Lovasz_loss(p3, label_batch)
            loss_iou4 = Lovasz_loss(p4, label_batch)
            loss_ohem_ce1 = ohem_ce_loss(p1, label_batch[:].long())
            loss_ohem_ce2 = ohem_ce_loss(p2, label_batch[:].long())
            loss_ohem_ce3 = ohem_ce_loss(p3, label_batch[:].long())
            loss_ohem_ce4 = ohem_ce_loss(p4, label_batch[:].long())
            loss_p1 = alpha * loss_ohem_ce1 + (1-alpha) * loss_iou1
            loss_p2 = alpha * loss_ohem_ce2 + (1-alpha) * loss_iou2
            loss_p3 = alpha * loss_ohem_ce3 + (1-alpha) * loss_iou3
            loss_p4 = alpha * loss_ohem_ce4 + (1-alpha) * loss_iou4

            loss = loss_p1 +  loss_p2 +  loss_p3 + loss_p4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)


        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (
        iter_num, epoch_num, loss.item(), lr_))
        image = image_batch[1, 0:1, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        writer.add_image('train/Image', image, iter_num)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
        labs = label_batch[1, ...].unsqueeze(0) * 50
        writer.add_image('train/GroundTruth', labs, iter_num)

        # logging.info('iteration %d, epoch %d : loss : %f, lr: %f, weights: %s' % (iter_num, epoch_num, loss.item(), lr_,weights))
        if (epoch_num >= 150) and (epoch_num % args.save_interval) == 0:
            performance = inference(args, model, best_performance)
            model.train()
        else:
            performance = 0.0


        # if (best_performance <= performance) and (epoch_num >= 450):
        #     best_performance = performance
        #     save_mode_path = os.path.join(snapshot_path, 'best.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if (epoch_num + 1) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }

    args.exp = args.model_name +'_' + args.dataset
    snapshot_path = args.output_dir + args.exp
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrained else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+ str(args.seed) if args.seed != 1234 else snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    net = LUCF_Net(in_chns=args.in_chans, class_num=args.num_classes).cuda()


    if args.is_pretrained:
        net.load_state_dict(torch.load(args.pretrained_pth))

    trainer = {'Synapse': trainer_synapse, }
    trainer[args.dataset](args, net, snapshot_path)


