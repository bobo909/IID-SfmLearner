from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()