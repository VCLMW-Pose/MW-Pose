# -*- coding: utf-8 -*-
'''
    Created on Sat Nov 3 21:39 2018

    Author           : Shaoshu Yang
    Email            : shaoshuyangseu@gmail.com
    Last edit date   : Tues Nov 6 24:00 2018

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import grad
from src.model import DRAGAN
from src.dataset import MNIST
from src.utils import logger, imwrite

def train(args):
    # Test set
    z_test_set = torch.rand((args.batch_size, args.z_dim))
    z_sample = torch.rand((args.test_iter, args.z_dim))

    # Set training hyper parameters
    loss_min = float('inf')
    lambda_ = args.lambda_
    k = args.k
    epoches = args.epoch
    gpu_mode = args.gpu_mode
    GAN = DRAGAN(args.save_dir, args.model_name, args.img_size, args.z_dim, args.x_dim)
    mnist = DataLoader(MNIST(args.dataset, 20), batch_size=args.batch_size, shuffle=True, drop_last=True)
    mnist_test = DataLoader(MNIST(args.testset, 20), batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Resume
    if args.resume:
        GAN.load()

    # Set logger
    log = logger(args.log_dir, args.model_name, args.resume)
    log_test = logger(args.test_log_dir, args.model_name + '_test', args.resume)
    loss_tag= ['G_loss', 'D_loss', 'total']
    log.set_tags(loss_tag)
    log_test.set_tags(loss_tag)

    # Set optimizer
    G_optimizer = optim.Adam(GAN.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(GAN.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    # Set loss function
    BCEloss = nn.BCELoss()

    # Label for disciminator
    y_real, y_fake = torch.ones(args.batch_size, 1), torch.zeros(args.batch_size, 1)
    if gpu_mode:
        BCEloss.cuda()
        y_real, y_fake = y_real.cuda(), y_fake.cuda()
        GAN.G.cuda()
        GAN.D.cuda()
        z_test_set = z_test_set.cuda()
        z_sample = z_sample.cuda()
    if args.benchmark_mode:
        cudnn.benchmark = True

    # Training process
    for epoch in range(epoches):
        GAN.G.train()
        GAN.D.train()
        for i, x in enumerate(mnist):
            # Losses recorder
            losses = []

            # Prepare input
            z = torch.rand((args.batch_size, args.z_dim))
            if gpu_mode:
                x = x.cuda()
                z = z.cuda()

            G_optimizer.zero_grad()
            D_optimizer.zero_grad()

            # Loss of discriminator
            D_real = GAN.D(x)
            D_real_loss = BCEloss(D_real, y_real)

            G = GAN.G(z)
            D_fake = GAN.D(G)
            D_fake_loss = BCEloss(D_fake, y_fake)

            # Gradient penalty
            alpha = torch.rand(args.batch_size, 1, 1, 1)
            if gpu_mode:
                x_p = x + 0.5*x.std()*torch.rand(x.size()).cuda()
            else:
                x_p = x + 0.5 * x.std() * torch.rand(x.size())
            if gpu_mode:
                alpha = alpha.cuda()
            difference = x_p - x
            interpolates = x + (alpha*difference)
            interpolates.requires_grad = True
            pred_hat = GAN.D(interpolates)
            if gpu_mode:
                gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]

            else:
                gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = lambda_*((gradients.view(gradients.size()[0], -1).norm(2, 1) - k)**2).mean()

            # Back propagation for discriminator
            D_loss = D_real_loss + D_fake_loss + gradient_penalty
            D_loss.backward()
            D_optimizer.step()

            # Train generator
            G = GAN.G(z)
            D_fake = GAN.D(G)
            G_loss = BCEloss(D_fake, y_real)
            G_loss.backward()
            G_optimizer.step()

            # Upgrade logger
            losses.append(D_loss.item())
            losses.append(G_loss.item())
            losses.append(D_loss.item() + G_loss.item())
            log.append(losses)

            # Output training info
            print('[Epoch %d/%d, Batch %d/%d] [Losses: G_loss %f, D_loss %f, total %f]' %(epoch, epoches, i + 1, len(mnist),
                                                                                                       losses[0], losses[1],
                                                                                            D_loss.item() + G_loss.item()))

        # Validation
        #validate(mnist_test, GAN, z_test_set, loss_min, gpu_mode, BCEloss, y_real, y_fake, args.batch_size, lambda_, k,
        #                                                                                                      log_test)
        GAN.G.eval()
        GAN.D.eval()
        visualize_result(GAN, epoch, z_sample, args.result_dir)

    log.close()
    log_test.close()

def validate(testset, GAN, z_test, loss_min, gpu_mode, loss_function, y_real, y_fake, batch_size, lambda_, k, logger):
    GAN.G.eval()
    GAN.D.eval()
    total_loss = []

    # Validation process
    for i, x in enumerate(testset):
        losses = []
        if gpu_mode:
            x = x.cuda()

            # Loss of discriminator
            D_real = GAN.D(x)
            D_real_loss = loss_function(D_real, y_real)

            G = GAN.G(z_test)
            D_fake = GAN.D(G)
            D_fake_loss = loss_function(D_fake, y_fake)

            # Gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1)
            if gpu_mode:
                x_p = x + 0.5*x.std()*torch.rand(x.size()).cuda()
            else:
                x_p = x + 0.5 * x.std() * torch.rand(x.size())
            if gpu_mode:
                alpha = alpha.cuda()
            difference = x_p - x
            interpolates = x + (alpha * difference)
            interpolates.requires_grad = True
            pred_hat = GAN.D(interpolates)
            if gpu_mode:
                gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]

            else:
                gradients = grad(outputs=pred_hat, inputs=interpolates, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = lambda_ * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - k) ** 2).mean()

            # Back propagation for discriminator
            D_loss = D_real_loss + D_fake_loss + gradient_penalty

            # Loss of generator
            G = GAN.G(z_test)
            D_fake = GAN.D(G)
            G_loss = loss_function(D_fake, y_real)

            # Record loss
            losses.append(D_loss.item())
            losses.append(G_loss.item())
            losses.append(D_loss.item() + G_loss.item())
            logger.append(losses)
            total_loss.append(losses[2])

            # Output validation info
            print('[Validation, Batch %d/%d] [Losses: G_loss %f, D_loss %f, total %f]' % (i + 1, len(testset),
                                                                                            G_loss.item(), D_loss.item(),
                                                                                          G_loss.item() + D_loss.item()))
    if np.array(total_loss).mean() < loss_min:
        GAN.save()

def visualize_result(GAN, epoch, z_sample, result_dir):
    len = z_sample.shape[0]
    for i in range(len):
        img = GAN.G(z_sample[i].unsqueeze(0))
        imwrite(img[0], '../' + result_dir + '%d'% epoch + '_%05d'% i + '.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAGAN Training Experiment:")

    parser.add_argument('--dataset', default='D:/ShaoshuYang/MNIST/train', type=str, metavar='N', help='Directory '
                                                                                                        +'of data set')
    parser.add_argument('--testset', default='D:/ShaoshuYang/MNIST/test', type=str, metavar='N', help='Directory'
                                                                                                      +'of test set')
    parser.add_argument('--resume', default=False, type=bool, metavar='N', help='Resume traning or traning from '
                                                                                                        +'scratch')
    parser.add_argument('--model_name', default='DRAGAN', type=str, metavar='N', help='Name of model')
    parser.add_argument('--epoch', default=90, type=int, metavar='N', help='Epoches of training')
    parser.add_argument('--test_iter', default=10, type=int, metavar='N', help='Test set amount')
    parser.add_argument('--simple_num', default=100, type=int, metavar='N', help='Number of simples')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='Batch size of training')
    parser.add_argument('--save_dir', default='weights/', type=str, metavar='N', help='Directory of weight files')
    parser.add_argument('--result_dir', default='results/', type=str, metavar='N', help='Directory of result images')
    parser.add_argument('--log_dir', default='logger/', type=str, metavar='N', help='Directory to save logs')
    parser.add_argument('--test_log_dir', default='logger/', type=str, metavar='N', help='Directory to save test logs')
    parser.add_argument('--lrG', default=1e-4, type=float, metavar='N', help='Learning rate of generator')
    parser.add_argument('--lrD', default=1e-4, type=float, metavar='N', help='Learning rate of discriminator')
    parser.add_argument('--beta1', default=0.5, type=float, metavar='N')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='N')
    parser.add_argument('--lambda_', default=0.25, type=float, metavar='N')
    parser.add_argument('--k', default=1, type=float, metavar='N')
    parser.add_argument('--gpu_mode', default=True, type=bool, metavar='N', help='Whether use GPU or not')
    parser.add_argument('--benchmark_mode', default=True, type=bool, metavar='N', help='Whether use cudnn')
    parser.add_argument('--z_dim', default=100, type=int, metavar='N', help='Dimension of input scalar')
    parser.add_argument('--x_dim', default=3, type=int, metavar='N', help='Dimension of generated result')
    parser.add_argument('--img_size', default=20, type=int, metavar='N', help='Scale of generated result')

    args = parser.parse_args()

    train(args)