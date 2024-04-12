'''
Written by YYF.
'''
import argparse
from A.A_train import main_train as ATrain
from A.A_eval import main_eval as AEval
from A.A_testimg import show_testimg as ATest
from B.B_train import main_train as BTrain
from B.B_eval  import main_eval as BEval
from B.B_testimg import show_testimg as BTest

if __name__ == '__main__' :

    models_arr = ['SRCNN', 'SRResNet', 'SRGAN', 'EDSR']

    #for i in range(4):
    #    BTrain(models_arr[i], 3000)

    for i in range(4):
        ATrain(models_arr[i], 2500)


    parser = argparse.ArgumentParser(
        description='Processing DIV2K image data - Train Eval and Test')

    parser.add_argument('--task',
                        help="'A' or 'B', select A: Bicubic X2 or B: unknown data",
                        type=str,
                        required=True)

    parser.add_argument('--mode',
                        default='eval',  # mode is : train, eval, test
                        help="mode include train, eval and test.",
                        type=str)

    parser.add_argument('--model',
                        default='SRResNet',   #4 model: SRResNet, SRGAN, SRCNN, EDSR
                        help='include 4 model: SRResNet, SRGAN, SRCNN, EDSR',
                        type=str)

    parser.add_argument('--iter',
                        default=1000,   #iteration of training
                        help='iteration of training, related to epochs of training. about 3 epochs per 100 iters',
                        type=int)

    parser.add_argument('--serialno',
                        default='0823',   #test image serial number
                        help='test image serial number, 4digit string such as "0823" ',
                        type=str)

    parser.add_argument('--part1',
                        default=7,   #test image serial number
                        help='test image detailed part1, rang is :0~24 1/5*5 part of the image',
                        type=int)

    parser.add_argument('--part2',
                        default=68,   #test image serial number
                        help='test image detailed part2, rang is :0~99 1/10*10 part of the image',
                        type=int)


    args = parser.parse_args()
    taskid = args.task.lower()
    mode = args.mode.lower()
    model_name = args.model
    iter = args.iter
    serailno = args.serialno
    part1 = args.part1
    part2 = args.part2
    if taskid == 'a':
        if mode == 'train':
            ATrain(model_name, iter)
        elif mode == 'eval' :
            AEval(model_name)
        else:
            ATest(serailno, part1, part2)
    else:
        if mode == 'train':
            BTrain(model_name, iter)
        elif mode == 'eval' :
            BEval(model_name)
        else:
            BTest(serailno, part1, part2)
