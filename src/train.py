import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import config
from dataset import DNADataset
from model import Model
from utils import create_loaders, split_dataset


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def train(rank, world_size, model, optimizer, train_dataloader, val_dataloader, device):
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)

        total_train_loss = 0
        # TQDM progress bar only on the main process
        pbar = tqdm(
            train_dataloader,
            desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Training]',
            colour='#AA80FF',
            disable=(rank != 0)
        )
        for batch in pbar:
            batch = batch.to(device)
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Gather losses from all processes (optional, for accurate logging)
        # This part can be simplified if only rank 0 logs
        avg_train_loss = total_train_loss / len(train_dataloader)

        if rank == 0:
            avg_val_loss = evaluate(model, val_dataloader, 'Validation', device)
            print(
                f'Epoch {epoch+1}/{config.NUM_EPOCHS}, '
                f'train loss: {avg_train_loss:.4f}, '
                f'validation loss: {avg_val_loss:.4f}'
            )

        if world_size > 1:
            dist.barrier()


def evaluate(model, dataloader, mode, device):
    # Use model.module to get the original model when wrapped in DDP
    model_to_eval = model.module if isinstance(model, DistributedDataParallel) else model
    model_to_eval.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'[{mode}]', colour='#AA80FF'):
            batch = batch.to(device)

            loss = model_to_eval(batch)
            if loss is not None:
                total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def run(rank, world_size, dataset):
    setup(rank, world_size)
    try:
        device = rank

        train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
        train_dataloader, val_dataloader, test_dataloader = create_loaders(
            train_dataset, val_dataset, test_dataset, config.BATCH_SIZE, world_size=world_size, rank=rank
        )

        node_dim = dataset.num_node_features + 3
        edge_dim = 1

        model = Model(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=config.HIDDEN_DIM, num_timesteps=config.NUM_TIMESTEPS
        ).to(device)
        model = DistributedDataParallel(model, device_ids=[device])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

        train(rank, world_size, model, optimizer, train_dataloader, val_dataloader, device)

        if rank == 0:
            avg_test_loss = evaluate(model, test_dataloader, 'Test', device)
            print(f'Test loss: {avg_test_loss:.4f}')

            model_path = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'model.pth')
            # Save the state of the underlying model
            torch.save(model.module.state_dict(), model_path)
            print(f'Model saved to `{model_path}`')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f'This script is designed for multi-GPU training. Found {world_size} GPUs.')
        # Simple fallback or error could be here. For now, we assume user wants DDP.
        # To run on a single GPU, you might want to call a different main function.

    print(f'Spawning {world_size} processes.')
    # Load dataset once and share it across processes
    dataset = DNADataset()
    try:
        mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)  # type: ignore
    except KeyboardInterrupt:
        print('\nInterrupted by user. Cleaning up processes...')
