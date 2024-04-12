"""
Written by YYF.
"""

from Common.utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from Common.datasets import DIVDataset
from pathlib import Path


def main_eval( model_name = "SRResNet") :
    """
    main_eval: the eval the model.
    :param     model_name: model's name: "SRResNet", "SRGAN", "SRCNN", "EDSR"
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    BASEDIR = Path.cwd() # get the parent directory of the current file

    DATA_FOLDER = BASEDIR / 'Datasets'
    # test_data_names = ["Set5", "Set14", "BSDS100"]
    DESIRED_SIZE = None # size of target HR images

    SCALING_FACTOR = 2  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor


    # Model checkpoints
    TASK_ID = 'A'
    TASK_FOLDER = BASEDIR / "A" / "out"
    srresnet_ckpt = TASK_FOLDER / "A_ckpt_srresnet_best.pth.tar"
    srgan_ckpt = TASK_FOLDER / "A_ckpt_srgan_best.pth.tar"
    edsr_ckpt = TASK_FOLDER / "A_ckpt_edsr_best.pth.tar"
    srcnn_ckpt = TASK_FOLDER / "A_ckpt_srcnn_best.pth.tar"

    model_to_CKPT = {
        "SRResNet"  :   srresnet_ckpt,
        "SRGAN"     :   srgan_ckpt,
        "EDSR"      :   edsr_ckpt,
        "SRCNN"     :   srcnn_ckpt,
    }

    info_file = TASK_FOLDER / f"A_{model_name}_eval.txt"

    # Load model, either the SRResNet or the SRGAN

    file = open(info_file, "w")
    str = f"======== Task {TASK_ID} : Model: {model_name}  Evaling ... ========"
    print(str)
    file.write(f'{str}\n')

    if model_name != "SRGAN":
        net = torch.load(model_to_CKPT[model_name])['model'].to(device)
        net.eval()
        model = net
    else :
        srgan_generator = torch.load(srgan_ckpt)['generator'].to(device)
        srgan_generator.eval()
        model = srgan_generator

    # Evaluate


    if model_name == "SRResNet":
        # Custom dataloader
        test_dataset = DIVDataset(
            DATA_FOLDER,
            split="test",
            process="crop",
            desired_size=DESIRED_SIZE,
            scaling_factor=SCALING_FACTOR,
            lr_img_type="[0, 1]",
            hr_img_type="[0, 1]",
            task_id=TASK_ID)
    else:
        test_dataset = DIVDataset(
            DATA_FOLDER,
            split="test",
            process="crop",
            desired_size=DESIRED_SIZE,
            scaling_factor=SCALING_FACTOR,
            lr_img_type="[0, 1]",
            hr_img_type="[0, 1]",
            task_id=TASK_ID)


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                  pin_memory=True)

        # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

        # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

            '''
            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                            data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                             data_range=255.)
            '''
            if model_name == "SRResNet" or model_name == "SRGAN" :
                psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[-1, 1]', '[0, 1]')
            else:
                psnr, ssim = compute_metrics(sr_imgs, hr_imgs, '[0, 1]', '[0, 1]')

            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

    # Print average PSNR and SSIM
    str = 'PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs)
    print(str)
    file.write(f'{str}\n')

    str = 'SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs)
    print(str)
    file.write(f'{str}\n')
    str = f"======== End Task {TASK_ID} : Model: {model_name}  Eval ========"
    print(str)
    file.write(f'{str}\n')
    file.close()

if __name__ == '__main__':
    main_eval("SRResNet")
    main_eval("SRGAN")
    main_eval("EDSR")
    main_eval("SRCNN")




