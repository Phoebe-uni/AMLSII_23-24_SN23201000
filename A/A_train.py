"""
Written by YYF.
"""

import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from Common.models import SRResNet, Generator, Discriminator, TruncatedVGG19, RCAN, EDSR, SRCNN
from Common.datasets import DIVDataset
from Common.utils import *
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import os


def main_train(model_name = "SRResNet", iterations = 1000):
    """
    Training the model
    :param img: model_name: "SRResNet", "SRGAN", "SRCNN" or "EDSR"
    :param iterations:  the iterations of train data, it related to training epochs
    """
    best_psnr = 0.0
    best_ssim = 0.0

    # Data parameters
    BASEDIR = Path.cwd()  # get the parent directory of the current file
    DATA_FOLDER = BASEDIR / 'Datasets'
    DESIRED_SIZE = 96  # size of target HR images
    SCALING_FACTOR = 2  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
    TRAIN_SPLIT = 0.8  # 20% of the dataset will be used for validation

    # Model parameters
    large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
    small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
    n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
    n_blocks = 16  # number of residual blocks

    # Discriminator parameters
    #for SRGAN parameters
    kernel_size_d = 3  # kernel size in all convolutional blocks
    n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
    n_blocks_d = 8  # number of convolutional blocks
    large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
    small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
    n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
    n_blocks_g = 16  # number of residual blocks
    fc_size_d = 1024  # size of the first fully connected layer

    TASK_ID = 'A'
    TASK_FOLDER = BASEDIR / "A" / "out"
    srresnet_ckpt = TASK_FOLDER / "A_ckpt_srresnet.pth.tar"
    srgan_ckpt = TASK_FOLDER / "A_ckpt_srgan.pth.tar"
    srcnn_ckpt = TASK_FOLDER / "A_ckpt_srcnn.pth.tar"
    edsr_ckpt = TASK_FOLDER / "A_ckpt_edsr.pth.tar"

    ## save the best pnsr checkpoint to file.
    best_srresnet_ckpt = TASK_FOLDER / "A_ckpt_srresnet_best.pth.tar"
    best_srgan_ckpt = TASK_FOLDER / "A_ckpt_srgan_best.pth.tar"
    best_srcnn_ckpt = TASK_FOLDER / "A_ckpt_srcnn_best.pth.tar"
    best_edsr_ckpt = TASK_FOLDER / "A_ckpt_edsr_best.pth.tar"


    info_file = TASK_FOLDER / f"A_{model_name}_train.txt"
    fig_head = TASK_FOLDER / f"A_{model_name}"
    title_head = f"Task A: {model_name}"

    model_to_CKPT = {
        "SRResNet"  :   srresnet_ckpt,
        "SRGAN"     :   srgan_ckpt,
        "EDSR"      :   edsr_ckpt,
        "SRCNN"     :   srcnn_ckpt,

    }

    model_to_BestCKPT = {
        "SRResNet"  :   best_srresnet_ckpt,
        "SRGAN"     :   best_srgan_ckpt,
        "EDSR"      :   best_edsr_ckpt,
        "SRCNN"     :   best_srcnn_ckpt,

    }


    model_to_Function = {
        "SRResNet": SRResNet,
        "SRGAN": Generator,
        "EDSR": EDSR,
        "SRCNN": SRCNN,

    }

    #checkpoint = model_to_CKPT[model_name]
    checkpoint = None  # path to model checkpoint, None if none
    #best_checkpoint = model_to_BestCKPT[model_name]
    best_checkpoint = None  # path to best model checkpoint, None if none


    # Learning parameters
    BATCH_SIZE = 16  # batch size
    start_epoch = 0  # start at this epoch
    #iterations = 500  # 1e3  # number of training iterations
    WORKERS = 4  # number of workers for loading data in the DataLoader
    print_freq = 500  # print training status once every __ batches
    lr = 1e-4  # learning rate
    grad_clip = None  # clip if gradients are exploding

    ## for SRGAN paras.
    vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
    vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
    beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss



    train_loss_list = []
    Training_PSNR_list = []
    Training_SSIM_list = []
    Validation_PSNR_list = []
    Validation_SSIM_list = []
    valid_loss_list = []



    # Default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True

    # Initialize model or load checkpoint

    file = open(info_file, "w")
    str = f"======== Task {TASK_ID} : Model: {model_name}  Training ... ========"
    print(str)
    file.write(f'{str}\n')




    if checkpoint is None:
        if model_name != "SRGAN" :
            if model_name == "SRResNet" :
                model = model_to_Function[model_name](large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                                 n_channels=n_channels, n_blocks=n_blocks, scaling_factor=SCALING_FACTOR)
            else :
                model = model_to_Function[model_name]()
            # Initialize the optimizer
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=lr)
        else :
            # Generator
            generator = Generator(large_kernel_size=large_kernel_size_g,
                                  small_kernel_size=small_kernel_size_g,
                                  n_channels=n_channels_g,
                                  n_blocks=n_blocks_g,
                                  scaling_factor=SCALING_FACTOR)

            # Initialize generator network with pretrained SRResNet
            #print(srresnet_ckpt)
            generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_ckpt)

            # Initialize generator's optimizer
            optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                           lr=lr)

            # Discriminator
            discriminator = Discriminator(kernel_size=kernel_size_d,
                                          n_channels=n_channels_d,
                                          n_blocks=n_blocks_d,
                                          fc_size=fc_size_d)

            # Initialize discriminator's optimizer
            optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                           lr=lr)

    else:   ## read the checkpoint pth file and continue training...
        if model_name != "SRGAN" :
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            if best_checkpoint is not None:
                best_checkpoint = torch.load(best_checkpoint)
                model = best_checkpoint['model']
                optimizer = best_checkpoint['optimizer']
                best_psnr = best_checkpoint['psnr']
                best_ssim = best_checkpoint['ssim']

        else:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            generator = checkpoint['generator']
            discriminator = checkpoint['discriminator']
            optimizer_g = checkpoint['optimizer_g']
            optimizer_d = checkpoint['optimizer_d']
            if best_checkpoint is not None:
                best_checkpoint = torch.load(best_checkpoint)
                generator = checkpoint['generator']
                discriminator = checkpoint['discriminator']
                optimizer_g = checkpoint['optimizer_g']
                optimizer_d = checkpoint['optimizer_d']
                best_psnr = best_checkpoint['psnr']
                best_ssim = best_checkpoint['ssim']

        str = "\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1)
        print(str)
        file.write(f'{str}')

        if best_checkpoint is not None:
            str = "Loaded the best checkpoint. which psnr: {psnr:.4f}, ssim: {ssim:.4f}\n".format(psnr = best_psnr,
                    ssim = best_ssim)
            print(str)
            file.write(f'{str}')

    if model_name != "SRGAN":
        # Move to default device
        model = model.to(device)
        criterion = nn.MSELoss().to(device)
    else :
        # Truncated VGG19 network to be used in the loss calculation
        truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
        truncated_vgg19.eval()

        # Loss functions
        content_loss_criterion = nn.MSELoss()
        adversarial_loss_criterion = nn.BCEWithLogitsLoss()

        # Move to default device
        generator = generator.to(device)
        model = generator
        discriminator = discriminator.to(device)
        truncated_vgg19 = truncated_vgg19.to(device)
        content_loss_criterion = content_loss_criterion.to(device)
        adversarial_loss_criterion = adversarial_loss_criterion.to(device)
        criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    if model_name == "SRResNet":
        whole_dataset = DIVDataset(
        DATA_FOLDER,
        split="train",
        process="crop",
        desired_size=DESIRED_SIZE,
        scaling_factor=SCALING_FACTOR,
        lr_img_type="[0, 1]",
        hr_img_type="[-1, 1]",
        task_id = TASK_ID )
    elif model_name == "SRGAN":
        whole_dataset = DIVDataset(
        DATA_FOLDER,
        split="train",
        process="crop",
        desired_size=DESIRED_SIZE,
        scaling_factor=SCALING_FACTOR,
        lr_img_type="[0, 1]",
        hr_img_type="[0, 1]",
        task_id = TASK_ID )
    else :
        whole_dataset = DIVDataset(
        DATA_FOLDER,
        split="train",
        process="crop",
        desired_size=DESIRED_SIZE,
        scaling_factor=SCALING_FACTOR,
        lr_img_type="[0, 1]",
        hr_img_type="[0, 1]",
        task_id = TASK_ID )


    # Calculate the sizes of train and validation sets
    train_size = int(TRAIN_SPLIT * len(whole_dataset))
    val_size = len(whole_dataset) - train_size

    # Divide the dataset into train and validation sets
    train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size])


    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)
    str = f'The number of epochs is {epochs}'
    print(str)
    file.write(f'{str}\n')

    # Epochs
    for epoch in range(start_epoch, epochs):
        if model_name != "SRGAN":
            # One epoch's training
            train_loss, Training_PSNR, Training_SSIM = Training_model(train_loader=train_loader,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch,
                  file = file, model_name = model_name)
            train_loss_list.append(train_loss)
            Training_PSNR_list.append(Training_PSNR)
            Training_SSIM_list.append(Training_SSIM)

        else:
            # At the halfway point, reduce learning rate to a tenth
            if epoch == int((iterations / 2) // len(train_loader) + 1):
                adjust_learning_rate(optimizer_g, 0.1)
                adjust_learning_rate(optimizer_d, 0.1)

            # One epoch's training
            train_loss, Training_PSNR, Training_SSIM = train_SRGAN(train_loader=train_loader,
                  generator=generator,
                  discriminator=discriminator,
                  truncated_vgg19=truncated_vgg19,
                  content_loss_criterion=content_loss_criterion,
                  adversarial_loss_criterion=adversarial_loss_criterion,
                  criterion=criterion,
                  optimizer_g=optimizer_g,
                  optimizer_d=optimizer_d,
                  epoch=epoch,
                  file = file )
            train_loss_list.append(train_loss)
            Training_PSNR_list.append(Training_PSNR)
            Training_SSIM_list.append(Training_SSIM)

        if Training_PSNR > best_psnr:    ## if training_psnr > best_pnsr, then save this epoch data to file.
            best_psnr = Training_PSNR
            best_ssim = Training_SSIM
            str = "****save the best pnsr checkpoint. epoch: {0}, psnr: {psnr:.3f}, ssim: {ssim:.3f}\n".format(epoch,
                   psnr=best_psnr,
                   ssim=best_ssim)
            print(str)
            file.write(str)
            if model_name != "SRGAN":
                # Save checkpoint
                torch.save({'epoch': epoch,
                            'psnr': best_psnr,
                            'ssim' : best_ssim,
                        'model': model,
                        'optimizer': optimizer},
                        model_to_BestCKPT[model_name])
            else :
                torch.save({'epoch': epoch,
                            'psnr': best_psnr,
                            'ssim' : best_ssim,
                            'generator': generator,
                            'discriminator': discriminator,
                            'optimizer_g': optimizer_g,
                            'optimizer_d': optimizer_d},
                           model_to_BestCKPT[model_name])

        if epoch % 10 == 0:
            # validation every 10 epochs
            valid_loss, Validation_PSNR, Validation_SSIM = valid(valid_loader=valid_loader,
              model=model,
              criterion=criterion,
              epoch=epoch, model_name = model_name, file = file)
            valid_loss_list.append(valid_loss)
            Validation_PSNR_list.append(Validation_PSNR)
            Validation_SSIM_list.append(Validation_SSIM)
            #also save the checkpoint pth to prevent the computer is shut down abnormally.
            if model_name != "SRGAN":
                # Save checkpoint
                torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer},
                        model_to_CKPT[model_name])
            else :
                torch.save({'epoch': epoch,
                            'generator': generator,
                            'discriminator': discriminator,
                            'optimizer_g': optimizer_g,
                            'optimizer_d': optimizer_d},
                        model_to_CKPT[model_name])

    if model_name != "SRGAN":
        # Save checkpoint
        torch.save({'epoch': epoch,
                'model': model,
                'optimizer': optimizer},
                model_to_CKPT[model_name])
    else :
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d},
                   model_to_CKPT[model_name])

    str = f"======== End Task {TASK_ID} : Model: {model_name}  Training ========"
    print(str)
    file.write(f'{str}\n')

    file.write("\n---Now save the Training Loss, PSNRS and SSIMs---\n")
    file.write("epoch, train_loss, psnr, ssim\n")
    for i in range(len(train_loss_list)):
        str = "{0}, {loss:.3f}, {psnr:.3f}, {ssim:.3f}\n".format(i+1, loss = train_loss_list[i],
                psnr = Training_PSNR_list[i], ssim = Training_SSIM_list[i])
        file.write(str)

    file.write("\n---Now save the Valid Loss, PSNRS and SSIMs---\n")
    file.write("No., train_loss, psnr, ssim\n")
    for i in range(len(valid_loss_list)):
        str = "{0}, {loss:.3f}, {psnr:.3f}, {ssim:.3f}\n".format(i+1, loss = valid_loss_list[i],
                psnr = Validation_PSNR_list[i], ssim = Validation_SSIM_list[i])
        file.write(str)

    file.close()

    ## plot the training loss, psnr, ssim, and valid loss, psnr and ssim.
    plt.figure(1)
    plt.plot([i for i in range(len(train_loss_list))], train_loss_list)
    plt.ylabel("Training loss")
    plt.xlabel('Number of epochs')
    plt.title(f"{title_head} Training loss")
    plt.savefig(f'{fig_head}_Training_loss.png')

    plt.figure(2)
    plt.plot([i for i in range(len(Training_PSNR_list))], Training_PSNR_list)
    plt.ylabel("Triaining PSNR")
    plt.xlabel('Number of epochs')
    plt.title(f"{title_head} Training PSNR")
    plt.savefig(f'{fig_head}_Training_PSNR.png')

    plt.figure(3)
    plt.plot([i*10 for i in range(len(valid_loss_list))], valid_loss_list)
    plt.ylabel("Validation loss")
    plt.xlabel('Number of epochs')
    plt.title(f"{title_head} Validation loss")
    plt.savefig(f'{fig_head}_Validation_loss.png')

    plt.figure(4)
    plt.plot([i*10 for i in range(len(Validation_PSNR_list))], Validation_PSNR_list)
    plt.ylabel("Validation PSNR")
    plt.xlabel('Number of epochs')
    plt.title(f"{title_head} Validation PSNR")
    plt.savefig(f'{fig_head}_Validation_PSNR.png')

    plt.figure(5)
    plt.plot([i for i in range(len(Training_SSIM_list))], Training_SSIM_list)
    plt.ylabel("Triaining SSIM")
    plt.xlabel('Number of epochs')
    plt.title(f"{title_head} Training SSIM")
    plt.savefig(f'{fig_head}_Training_SSIM.png')

    plt.figure(6)
    plt.plot([i for i in range(len(Validation_SSIM_list))], Validation_SSIM_list)
    plt.ylabel("Validation SSIM")
    plt.xlabel('Number of epochs')
    plt.title(f"{title_head} Validation SSIM")
    plt.savefig(f'{fig_head}_Validation_SSIM.png')
    plt.close("all")




def Training_model(train_loader, model, criterion, optimizer, epoch, file, model_name):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    :param file: save the print info to file
    :param model_name: the model_name
    """
    print_freq = 500  # print training status once every __ batches
    grad_clip = None  # clip if gradients are exploding

    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    train_losses = AverageMeter()  # training loss

    # Keep track of the PSNRs and the SSIMs across batches
    Training_PSNRs = AverageMeter()
    Training_SSIMs = AverageMeter()
    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

#        if model_name != "SRResNet":
#            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='[0, 1]')  # (N, 3, 96, 96), imagenet-normed

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar
        
        ## calc the psnr and ssim
        if model_name == "SRResNet":
            psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[-1, 1]', '[-1, 1]')
        else:
            psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[0, 1]', '[0, 1]')

        Training_PSNRs.update(psnr, lr_imgs.size(0))
        Training_SSIMs.update(ssim, lr_imgs.size(0))
        # Backward prop.
        optimizer.zero_grad()

        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        train_losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            str = ('Epoch: [{0}][{1}/{2}]----' + \
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----' + \
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----' + \
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})').format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=train_losses)
            print(str)
            file.write(f'{str}\n')

            str = 'Training PSNR - {psnrs.avg:.3f}'.format(psnrs=Training_PSNRs)
            print(str)
            file.write(f'{str}\n')
            str = 'Training SSIM - {ssims.avg:.3f}'.format(ssims=Training_SSIMs)
            print(str)
            file.write(f'{str}\n')

    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
    return train_losses.avg, Training_PSNRs.avg, Training_SSIMs.avg

def train_SRGAN(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          criterion, optimizer_g, optimizer_d, epoch, file):
    """
    One epoch's training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    :param file: save the print info to file
    """
    print_freq = 500  # print training status once every __ batches
    grad_clip = None  # clip if gradients are exploding
    beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss

    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator
    # Keep track of the PSNRs and the across batches

    train_losses = AverageMeter()  # training loss

    # Keep track of the PSNRs and the SSIMs across batches
    Training_PSNRs = AverageMeter()
    Training_SSIMs = AverageMeter()

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='[0, 1]')  # (N, 3, 96, 96), imagenet-normed

        ## calc the psnr and ssim
        psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[0, 1]', '[0, 1]')

        Training_PSNRs.update(psnr, lr_imgs.size(0))
        Training_SSIMs.update(ssim, lr_imgs.size(0))

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # detached because they're constant, targets

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()
        # Keep track of loss

        train_losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            '''
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))
            '''
            str = ('Epoch: [{0}][{1}/{2}]----' + \
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----' + \
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----' + \
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----' + \
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----' + \
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})').format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d)

            print(str)
            file.write(f'{str}\n')

            str = 'Training PSNR - {psnrs.avg:.3f}'.format(psnrs=Training_PSNRs)
            print(str)
            file.write(f'{str}\n')
            str = 'Training SSIM - {ssims.avg:.3f}'.format(ssims=Training_SSIMs)
            print(str)
            file.write(f'{str}\n')

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated  # free some memory since their histories may be stored
    return train_losses.avg, Training_PSNRs.avg, Training_SSIMs.avg


def valid(valid_loader, model, criterion, epoch, model_name, file):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    :model_name: the model_name
    :file: print info to file
    """
    print_freq = 500  # print training status once every __ batches

    model.eval()  # evaluation mode

    valid_losses = AverageMeter()  # loss
    # Keep track of the PSNRs and the SSIMs across batches
    Validation_PSNRs = AverageMeter()
    Validation_SSIMs = AverageMeter()

    # start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(valid_loader):
        # data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        if model_name == "SRGAN" :
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='[0, 1]')  # (N, 3, 96, 96), imagenet-normed
        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        '''
        # Calculate PSNR 
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(),data_range=255.)
        ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().detach().numpy(), data_range=255.)
        '''
        if model_name == "SRResNet":
            psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[-1, 1]', '[-1, 1]')
        else:
            psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[0, 1]', '[0, 1]')


        Validation_PSNRs.update(psnr, lr_imgs.size(0))
        Validation_SSIMs.update(ssim, lr_imgs.size(0))


        # # Backward prop.
        # optimizer.zero_grad()
        # loss.backward()

        # # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        # # Update model
        # optimizer.step()

        # Keep track of loss
        valid_losses.update(loss.item(), lr_imgs.size(0))

        # # Keep track of batch time
        # batch_time.update(time.time() - start)

        # Reset start time
        # start = time.time()

        # Print status
        if i % print_freq == 0:

            #print('Epoch: [{0}][{1}/{2}]----'
            #      'Validation Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(valid_loader),
            #                                                        loss=valid_losses))

            str = 'Epoch: [{0}][{1}/{2}]----Validation Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i,
                        len(valid_loader), loss=valid_losses)
            print(str)
            file.write(f'{str}\n')

            str = 'Validation PSNR - {psnrs.avg:.3f}'.format(psnrs=Validation_PSNRs)
            print(str)
            file.write(f'{str}\n')
            str = 'Validation SSIM - {ssims.avg:.3f}'.format(ssims=Validation_SSIMs)
            print(str)
            file.write(f'{str}\n')

    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
    return valid_losses.avg, Validation_PSNRs.avg, Validation_SSIMs.avg


if __name__ == '__main__':
    main_train("SRCNN", 100)
    #main_train("SRCNN", 400)
