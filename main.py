import argparse
import logging
import os
import random

import numpy as np
import torch

from DataloaderConstructing import DataloaderConstructing
from LLMFew import LLMFew
from Trainer import Trainer


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Optional: Enable if using multi-GPU
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for training LLMFew model.")

    # Experiment setup
    parser.add_argument('--dataset', default='BasicMotions', help='Dataset to use for training and evaluation.')
    parser.add_argument('--few_shot', type=int, default=1, help='Whether to use few-shot learning scenario.')
    parser.add_argument('--ablation', type=str, default='train', help='Type of ablation study to perform.')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for random number generators.')
    parser.add_argument('--path', default='ckpt', help='Path to save checkpoints.')

    # Model architecture
    parser.add_argument('--dimensions', type=int, default=6, help='Number of input planes.')
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=4, help='Number of classes.')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the network.')
    parser.add_argument('--channels', type=int, default=256, help='Number of channels.')
    parser.add_argument('--reduced_size', type=int, default=128, help='Reduced size for bottleneck layers.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutions.')
    parser.add_argument('--patch_len', type=int, default=16, help='Length of each patch for embeddings.')
    parser.add_argument('--stride', type=int, default=8, help='Stride for patch embedding.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')

    # Training process
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save_checkpoint', action='store_true', help='Whether to save checkpoints during training.')
    parser.add_argument('--interval', type=int, default=2, help='Interval between tseting.')

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer.')
    parser.add_argument('--step_size', type=int, default=50, help='Step size for learning rate decay.')
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma for learning rate decay.')

    # Large language model settings
    parser.add_argument('--llm_type', default='llama3', help='Type of large language model to use.')
    parser.add_argument('--lora', type=int, default=1, help='Use Lora layers or not.')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed != -1:
        seed_everything(args.seed)
    else:
        seed = random.randint(1, 100)
        print(f"Random seed: {seed}")
        seed_everything(seed)

    train_loader, test_loader = DataloaderConstructing(args.dataset, batch_size=args.batch_size, few_shot=args.few_shot)

    model = LLMFew(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')
    
    # Get best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    model = model.to(device)
    
    trainer = Trainer(model, train_loader, test_loader, args)

    print(f"Training preparation finished! Start to train on the {args.dataset} dataset! Good Luck! :)")
    trainer.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
    torch.cuda.empty_cache()  # Ensures that all CUDA memory is freed up when training is complete
