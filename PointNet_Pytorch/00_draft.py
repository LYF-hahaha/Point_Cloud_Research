import time

import numpy as np
import torch
from torch.nn.functional import nll_loss
from tqdm import tqdm


def test():
    lt = ['a', 'b', 'c']
    for i, item in enumerate(tqdm(lt)):
        print(f"i={i}  item={item}")
        time.sleep(0.2)


def main():
    test()


if __name__ == "__main__":
    main()
