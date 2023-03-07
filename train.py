import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from functools import partial
from trainer import DGTrainer
from dataset import load_dataset, DialogueDataset, collate_fn
import os

def main(parser):
    data_loaders = dict()
    args = parser.parse_args()

    wandb.init(project = 'dialogue generation')

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False # cudnn은 convolution을 수행하는 과정에서 가장 적합한 알고리즘을 선정해 수행
    torch.backends.cudnn.determinitic = True # 
    torch.use_deterministic_algorithms(True)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    print(f'num_workers : {args.num_workers}')
    print(f'learning_rate : {args.learning_rate}')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                                pad_token = '<pad>',
                                                bos_token = '<s>',
                                                eos_token = '</s>',
                                                unk_token = '<unk>',
                                                mask_token = '<mask>',
                                                model_max_length = 1024)
    tokenizer.add_tokens(['<sep>'])

    # print(tokenizer['<pad>'])

    print(f'tokenizer.pad_token : {tokenizer.pad_token_id}')
    
    print(f'tokenizer.bos_token : {tokenizer.bos_token_id}')
    
    print(f'tokenizer.eos_token : {tokenizer.eos_token_id}')
    
    print(f'tokenizer.unk_token : {tokenizer.unk_token_id}')
    
    print(f'tokenizer.mask_token : {tokenizer.mask_token_id}')
    

    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.resize_token_embeddings(len(tokenizer)) # 변경된 토큰의 수만큼 임베딩을 변경
    model = nn.DataParallel(model).to(device)
    model = model.to(device)

    train_datasets = load_dataset(args.menu, args.train_dir, '<sep>', '</s>')
    train_dataset = DialogueDataset(train_datasets, tokenizer)
    valid_datasets = load_dataset(args.menu, args.valid_dir, '<sep>', '</s>')
    valid_dataset = DialogueDataset(valid_datasets, tokenizer)

    print(f"batch_size : {args.batch_size}")

    data_loaders['train'] = DataLoader(dataset = train_dataset,
                                        num_workers=args.num_workers,
                                        collate_fn=partial(collate_fn, pad_token_id = tokenizer.pad_token_id, bos_token_id = tokenizer.bos_token_id ),
                                        batch_size=args.batch_size,
                                        drop_last=False)
    
    data_loaders['valid'] = DataLoader(dataset = valid_dataset,
                                        num_workers=args.num_workers,
                                        collate_fn=partial(collate_fn, pad_token_id = tokenizer.pad_token_id, bos_token_id = tokenizer.bos_token_id ),
                                        batch_size=args.batch_size,
                                        drop_last=False)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                    max_lr = args.learning_rate,
                                                    epochs=args.num_epochs,
                                                    steps_per_epoch=len(data_loaders['train']),
                                                    anneal_strategy='linear'
    )
    
    DGTrainer(
        model = model,
        train_loader = data_loaders['train'],
        valid_loader = data_loaders['valid'],
        optimizer = optimizer,
        scheduler = scheduler,
        num_epochs = args.num_epochs,
        device = device,
        tokenizer = tokenizer,
        gradient_clip_val = args.gradient_clip_val,
        log_every = args.log_every,
        save_every = args.save_every,
        save_dir = args.save_dir,
        accumulate_grad_batches = args.accumulate_grad_batches
    ).fit()
    



    return datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_model', type=str, default='skt/kogpt2-base-v2')
    parser.add_argument('-t', '--train_dir', type = str, default='/content/drive/MyDrive/gpt2/한국어 SNS/Training/[라벨]한국어SNS_train/')
    parser.add_argument('-v', '--valid_dir', type = str, default='/content/drive/MyDrive/gpt2/한국어 SNS/Validation/[라벨]한국어SNS_valid/')
    parser.add_argument('-d', '--save_dir', type = str, default='/content/drive/MyDrive/gpt2_checkpoints/checkpoints')
    parser.add_argument('-w', '--num_workers', type = int, default=4)
    parser.add_argument('-b', '--batch_size', type = int, default=8)
    parser.add_argument('-l', '--learning_rate', type = float, default=3e-5)
    parser.add_argument('-e', '--num_epochs', type = int, default=5)
    parser.add_argument('-g', '--gradient_clip_val', type = float, default=1.0 )
    parser.add_argument('-o', '--log_every', type=int, default=20)
    parser.add_argument('-a', '--accumulate_grad_batches', type = int, default=1)
    parser.add_argument('-s', '--save_every', type=int, default=10_000)
    parser.add_argument('-m', '--menu', type=str, default='kakao')

    datasets = main(parser)
