"""
Written by YYF.
"""

import torch
from Common.utils import *
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
from Common.models import SRResNet, Generator, Discriminator, TruncatedVGG19, RCAN, EDSR, SRCNN

#def visualize_sr(lr_img, hr_img, halve=False):
def show_testimg(serialno = "0010", rectno1 = 6, rectno2 = 66):

    """
    Show the testimg. we will show the HR img, LR img, SRCNN, SRResNet, SRGAN and EDSR processed img

    :param serialno: the serialno image. 4digit,such as "0064"
    :param halve:
    :rectno1:   we will show the part detailed. first part is a part of 1/(5*5). so rectno1 is a sequence no from upper
                to bottom, left to right. value is 0~24. eg.  7 is the second row, and third col.
    :rectno2:   the second detailed part is 1/(10*10). value is 0~99, so 45 is  the 5th row, 6th col.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data
    BASEDIR = Path.cwd() # get the parent directory of the current file
    DATA_FOLDER = BASEDIR / 'Datasets'
    sno = int(serialno)
    if sno > 800 :
        lr_dir = DATA_FOLDER / "DIV2K_valid_LR_bicubic_X2" / "DIV2K_valid_LR_bicubic" / "X2" /f"{serialno}x2.png"
        hr_dir = DATA_FOLDER / "DIV2K_valid_HR" / f"{serialno}.png"
    else:
        lr_dir = DATA_FOLDER / "DIV2K_train_LR_bicubic_X2" / "DIV2K_train_LR_bicubic" / "X2" /f"{serialno}x2.png"
        hr_dir = DATA_FOLDER / "DIV2K_train_HR" / f"{serialno}.png"

    ##checkpoint pth file.
    TASK_FOLDER = BASEDIR / "A" / "out"


    srresnet_ckpt = TASK_FOLDER / "A_ckpt_srresnet_best.pth.tar"
    srgan_ckpt = TASK_FOLDER / "A_ckpt_srgan_best.pth.tar"
    edsr_ckpt = TASK_FOLDER / "A_ckpt_edsr_best.pth.tar"
    srcnn_ckpt = TASK_FOLDER / "A_ckpt_srcnn_best.pth.tar"

    '''
    srresnet_ckpt = TASK_FOLDER / "A_ckpt_srresnet.pth.tar"
    srgan_ckpt = TASK_FOLDER / "A_ckpt_srgan.pth.tar"
    edsr_ckpt = TASK_FOLDER / "A_ckpt_edsr.pth.tar"
    srcnn_ckpt = TASK_FOLDER / "A_ckpt_srcnn.pth.tar"
    '''
    '''
    srresnet_ckpt = TASK_FOLDER / "B_ckpt_srresnet_best.pth.tar"
    srgan_ckpt = TASK_FOLDER / "B_ckpt_srgan_best.pth.tar"
    edsr_ckpt = TASK_FOLDER / "B_ckpt_edsr_best.pth.tar"
    srcnn_ckpt = TASK_FOLDER / "B_ckpt_srcnn_best.pth.tar"
    '''




    fig_file = TASK_FOLDER / f"A_TestImg({serialno}).png"

    models_arr  = ['SRCNN', 'SRResNet', 'SRGAN', 'EDSR']
    model_to_CKPT = {
        "SRResNet"  :   srresnet_ckpt,
        "SRGAN"     :   srgan_ckpt,
        "EDSR"      :   edsr_ckpt,
        "SRCNN"     :   srcnn_ckpt,
    }
    model_to_Function = {
        "SRResNet": SRResNet,
        "SRGAN": Generator,
        "EDSR": EDSR,
        "SRCNN": SRCNN,
    }
    model_to_ckptmodel = {
        "SRResNet": "model",
        "SRGAN": "generator",
        "EDSR": "model",
        "SRCNN": "model",
    }


    img_list = []
    title_list = []
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(hr_dir, mode="r")
    hr_img = hr_img.convert('RGB')

    lr_img = Image.open(lr_dir, mode="r")
    lr_img = lr_img.convert('RGB')

    hr_img = hr_img.resize((int(hr_img.width // 2), int(hr_img.height // 2)),
                           Image.BICUBIC)

    iscrop = 0
    if hr_img.width > 950 and hr_img.height > 950 :
        iscrop = 1
        height1 = hr_img.width *3//4
        if height1%2 == 1:
            height1 = height1 -1
        hr_img = hr_img.crop((0,0,hr_img.width, height1))

    # because my gpu is gtx1660, we have to resize the whole lr_img to half, that the gpu can process it.
    if iscrop == 1 :
        lr_img = lr_img.crop((0,0,lr_img.width, lr_img.width *3//4))

    lr_img = lr_img.resize((int(hr_img.width // 2), int(hr_img.height // 2)), Image.BICUBIC)

    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    img_list.append(hr_img)
    str = f"HiRes({serialno})"
    title_list.append(str)
    img_list.append(bicubic_img)
    psnr = compute_psnr(bicubic_img, hr_img)
    ssim = compute_ssim(bicubic_img, hr_img)

    str ="LowRes PSNR: %.3f SSIM: %.3f" % (psnr, ssim)
    title_list.append(str)

    for i in range(4) :
        model = torch.load(model_to_CKPT[models_arr[i]])[model_to_ckptmodel[models_arr[i]]].to(device)
        model.eval()
        sr_img = model(convert_image(lr_img, source='pil', target='[0, 1]').unsqueeze(0).to(device))
        if models_arr[i] == "SRResNet" or models_arr[i] == "SRGAN":
            sr_img = torch.clip(sr_img, -1, 1)
        else :
            sr_img = torch.clip(sr_img, 0, 1)
        sr_img = sr_img.squeeze(0).cpu().detach()

        if models_arr[i] == "SRResNet" or models_arr[i] == "SRGAN":
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        else :
            sr_img = convert_image(sr_img, source='[0, 1]', target='pil')

        img_list.append(sr_img)
        #print(f"sr_img: ({sr_img.width}, {sr_img.height}),  hr_img:({hr_img.width}, {hr_img.height})\n")
        psnr = compute_psnr(sr_img, hr_img)
        ssim = compute_ssim(sr_img, hr_img)
        str = "%s PSNR: %.3f SSIM: %.3f"%(models_arr[i], psnr, ssim)

        title_list.append(str)


    upper_margin = 160
    margin = 100
    rect1_x = rectno1 % 5
    rect1_y = rectno1 // 5

    rect2_x = rectno2 % 10
    rect2_y = rectno2 // 10

    grid_img = Image.new('RGB', (6 * hr_img.width + 7 * margin, 3 * hr_img.height + 3 * margin + upper_margin),
                         (255, 255, 255))
    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("c:/Windows/Fonts/Arial.ttf", size=60)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    for i in range(6) :
        str = title_list[i]
        x0, y0, x1, y1 = font.getbbox(str)
        text_size = (x1 - x0, y1 - y0)
        draw.text(xy=[margin*(i+1) + hr_img.width*i + hr_img.width / 2 - text_size[0] / 2, upper_margin - text_size[1] - 20],
                  text=str,
                  font=font,
                  fill='black')
        grid_img.paste(img_list[i], (margin*(i+1) + hr_img.width *i, upper_margin))

    for i in range(6):
        x1 = hr_img.width * rect1_x // 5
        x2 = hr_img.width * (rect1_x +1)//5 - 1
        y1 = hr_img.height * rect1_y // 5
        y2 = hr_img.height * (rect1_y+1) // 5 - 1
        draw.rectangle([(hr_img.width * i + margin *(i+1) + x1, upper_margin +  y1 ),
                        (hr_img.width * i + margin *(i+1) + x2, upper_margin +  y2 )],
                         fill =None, outline ="red",width =4)
        img1 = img_list[i].crop((x1,y1, x2,y2))
        img1 = img1.resize((hr_img.width, hr_img.height), Image.LANCZOS)
        grid_img.paste(img1,  (margin*(i+1) + hr_img.width *i,
                               upper_margin + hr_img.height + margin))


        x1 = hr_img.width * rect2_x // 10
        x2 = hr_img.width * (rect2_x +1)//10 - 1
        y1 = hr_img.height * rect2_y // 10
        y2 = hr_img.height * (rect2_y+1) // 10 - 1
        draw.rectangle([(hr_img.width * i + margin *(i+1) + x1, upper_margin +  y1 ),
                        (hr_img.width * i + margin *(i+1) + x2, upper_margin +  y2 )],
                         fill =None, outline ="blue",width =4)
        img1 = img_list[i].crop((x1,y1, x2,y2))
        img1 = img1.resize((hr_img.width, hr_img.height), Image.LANCZOS)
        grid_img.paste(img1,  (margin*(i+1) + hr_img.width *i,
                               upper_margin + (hr_img.height + margin)*2))



    grid_img.save(fig_file)
    grid_img.close()
    return grid_img


if __name__ == '__main__':
    grid_img = show_testimg("0010", 12, 85)
    grid_img = show_testimg("0180", 12, 33)
    grid_img = show_testimg("0267", 17, 33)
    grid_img = show_testimg("0826", 7, 43)
    grid_img = show_testimg("0886", 6, 54)
    grid_img = show_testimg("0855", 12, 56)
