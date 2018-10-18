from Solver import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--net', default='AlexNet', type=str, help='net type')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=2, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()
