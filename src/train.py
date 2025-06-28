import os
import os.path as osp
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import config
from dataset import DNADataset
from model import Model
from utils import create_loaders, split_dataset


def evaluate(model, dataloader, device):
    # Use model.module to get the original model when wrapped in DDP
    model_to_eval = model.module if isinstance(model, DistributedDataParallel) else model
    model_to_eval.eval()

    batch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            loss = model_to_eval(batch)
            if loss is not None:
                batch_loss += loss.item()

    avg_loss = batch_loss / len(dataloader)
    return avg_loss


def train(rank, world_size, model, optimizer, train_dataloader, val_dataloader, device, writer):
    # TQDM progress bar only on the main process
    pbar = tqdm(
        range(config.NUM_EPOCHS),
        desc='Training',
        colour='#AA80FF',
        disable=(rank != 0)
    )
    for epoch in pbar:
        model.train()
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)

        batch_train_loss = 0

        for batch in train_dataloader:
            batch = batch.to(device)

            loss = model(batch)
            batch_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = batch_train_loss / len(train_dataloader)

        if rank == 0:
            avg_val_loss = evaluate(model, val_dataloader, device)

            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        if world_size > 1:
            dist.barrier()


def run(rank, world_size, dataset, log_dir):
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '50000'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        device = rank

        writer = None
        if rank == 0:
            writer = SummaryWriter(log_dir=log_dir)

        train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
        train_dataloader, val_dataloader, test_dataloader = create_loaders(
            train_dataset, val_dataset, test_dataset, config.BATCH_SIZE, world_size=world_size, rank=rank
        )

        if rank == 0:
            # Save test dataset to file for visualization in jupyter notebook
            test_data_list = list(test_dataset)  # type: ignore
            test_dataset_path = osp.join(log_dir, 'test_dataset.pt')
            torch.save(test_data_list, test_dataset_path)

        node_dim = dataset.num_node_features + 3
        edge_dim = 1
        model = Model(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=config.HIDDEN_DIM, num_timesteps=config.NUM_TIMESTEPS
        ).to(device)
        model = DistributedDataParallel(model, device_ids=[device])
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

        train(rank, world_size, model, optimizer, train_dataloader, val_dataloader, device, writer)

        if rank == 0:
            avg_test_loss = evaluate(model, test_dataloader, device)

            if writer:
                hparams = {
                    'lr': config.LR,
                    'batch_size': config.BATCH_SIZE,
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_timesteps': config.NUM_TIMESTEPS,
                    'num_epochs': config.NUM_EPOCHS,
                    'bond_threshold': config.BOND_THRESHOLD,
                    'window_size': config.WINDOW_SIZE
                }
                metrics = {'Test Loss': avg_test_loss}
                writer.add_hparams(hparams, metrics)

            model_path = osp.join(log_dir, 'model.pth')
            torch.save(model.module.state_dict(), model_path)
    finally:
        if rank == 0 and writer:
            writer.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    # Create log directory with a human-readable name
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'logs', current_time)
    os.makedirs(log_dir)

    dataset = DNADataset(config.BOND_THRESHOLD, config.WINDOW_SIZE)

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    print(f'Spawning {world_size} processes.')

    try:
        mp.spawn(run, args=(world_size, dataset, log_dir), nprocs=world_size)  # type: ignore
    except KeyboardInterrupt:
        print('\nInterrupted by user. Cleaning up processes...')
