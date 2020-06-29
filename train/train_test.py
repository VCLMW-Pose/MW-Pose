#-*- coding = utf-8 -*-

import torch
import argparse
import os
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad
from src.utils.pose_decoder import *
from src.utils.evaluation import *
from src.model.deseqnet import DeSeqNetProj, DeSeqNetTest
from src.dataset import deSeqNetLoader
from src.utils import logger, imwrite
from src.model import BaseArch
from src.dataset import VGGLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument('--data_path', type=str, default="F:/datasets/origin", help="directory of dataset")
    #parser.add_argument("--pretrained_weights", type=str, default="checkpoints/deseqnettest_490.pth",
    # help="if specified starts from checkpoint model")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    with open(os.path.join(opt.data_path, 'keypoint.names'), 'r') as namefile:
        keypointnames = namefile.readlines()
        keypointnames = [line.rstrip() for line in keypointnames]

    logger_train = logger('logger.txt')
    logger_val = logger('logger_val.txt')
    logger_tag = keypointnames.copy()
    logger_tag.append('total')
    logger_valtag = keypointnames.copy()
    logger_valtag.append('average')
    logger_train.set_tags(logger_tag)
    logger_val.set_tags(logger_valtag)

    # Initiate model
    model = BaseArch("VGG").to(device)
    lossfunc = nn.MSELoss()
    if torch.cuda.is_available():
        lossfunc = lossfunc.cuda()

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        model.load_state_dict(torch.load(opt.pretrained_weights))

    # Get dataloader
    dataset = VGGLoader(opt.data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
    )

    validset = VGGLoader(opt.data_path, valid=True)
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (targets, signal, _) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            signal = Variable(signal.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # targets = torch.cat((targets, targets), 1)
            outputs = model(signal)
            losses = []
            accum_loss = 0
            # ----------------
            #   Train and Log progress
            # ----------------
            log_str = "[Epoch %d/%d, Batch %d/%d]" % (epoch, opt.epochs, batch_i, len(dataloader))

            for keypoint_i, name in enumerate(keypointnames):
                loss = lossfunc(outputs[:, keypoint_i, :, :], targets[:, keypoint_i, :, :])
                losses.append(loss.item())
                log_str += "[%s: %f]" % (name, loss)
                accum_loss = accum_loss + loss

            losses.append(accum_loss.item())
            logger_train.append(losses)
            log_str += "[total: %f]\n" % accum_loss
            print(log_str)
            accum_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        # Validation progress
        if epoch % opt.evaluation_interval == 0:
            model.eval()
            print("\n---- Evaluating Model ----\n")
            # Evaluate the model on the validation set

            # PCKh(8x18) stores the percentage of correct keypoint (head) during validation.
            # They are respectively     PCKh@0.05               PCKh@0.10
            #                           PCKh@0.20               PCKh@0.30
            #                           PCKh@0.40               PCKh@0.50
            #                           PCKh@0.75               PCKh@1.00
            n = len(keypointnames)
            PCKh = np.zeros([8, n + 1])
            thre = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]

            for batch_i, (_, val_signal, GT) in enumerate(validloader):

                val_signal = Variable(val_signal.to(device), requires_grad=False)
                GT = torch.cat((GT, GT), 0)
                val_outputs = model(val_signal)
                val_outputs = val_outputs.cpu()
                pred = pose_decode(val_outputs)
                print('[Valid, Batch %d/%d]\n' % (batch_i, len(validloader)))
                for i in range(8):
                    PCKh[i, 0:-1] = PCKh[i, 0:-1] + eval_pckh(pred, GT, n, thre[i])

            PCKh[:] = PCKh[:] / len(validloader)
            for i in range(8):
                PCKh[i, -1] = np.sum(PCKh[i, 0:-1]) / n
                logger_val.append(list(PCKh[i, :]))

                log_str = "PCKh@%f" % thre[i]
                for keypoint_i, name in enumerate(keypointnames):
                    log_str += '[%s, %f]' % (name, PCKh[i, keypoint_i])

                log_str += '[average, %f]\n' % PCKh[i, -1]
                print(log_str)

        if epoch % opt.checkpoint_interval == 9:
            torch.save(model.state_dict(), f"checkpoints/deseqnettest_%d.pth" % (epoch+1))

    logger_train.close()
    logger_val.close()