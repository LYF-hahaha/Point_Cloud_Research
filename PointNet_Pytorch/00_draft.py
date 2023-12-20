import numpy as np
import torch
from torch.nn.functional import nll_loss


def test():

    # c = [15, 33, 30,  5,  9, 36,  9, 35,  8,  8, 16,  8, 22,  0,  2,  1, 22,  0,
    #     26,  0,  4, 22,  5, 22, 30, 12,  9,  7,  5, 37, 18, 17]
    # d = torch.tensor(c)
    # e = d[:, 0]
    # print(c)
    # print(b)
    predict = torch.Tensor([[2, 3, 1],
                            [3, 7, 9]])
    label = torch.tensor([1, 2])
    result = nll_loss(predict, label)

    print(f"\nThe pred is:{predict}")
    print(f"\nThe label is:{label}")
    print(f"\nThe result is:{result}")


def main():
    test()


if __name__ == "__main__":
    main()
