import argparse
import json
import logging
import math
import random
from datetime import datetime
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.dataset import PairsDataset, TripletsDataset
from datasets.collate import collate_pairs_train, collate_triplets
from datasets.factory import create_dataloader
from model import CitRecFunModel
from utils import Config, mrr


def triplet_loss(scores, margin=None, final_loss_only=True, device='cpu'):
    margin = (torch.ones(scores[0].shape) * margin).to(device)
    zeros = torch.zeros(scores[0].shape).to(device)
    sem_loss = torch.max(scores[1] - scores[0] + margin, zeros)
    if len(scores) == 2:    # semantic model
        return torch.mean(sem_loss)
    else:                   # dual model
        if final_loss_only:
            loss = torch.max(scores[5] - scores[4] + margin, zeros)
        else:
            meta_loss = torch.max(scores[3] - scores[2] + margin, zeros)
            total_loss = torch.max(scores[5] - scores[4] + margin, zeros)
            loss = sem_loss + meta_loss + total_loss
        return torch.mean(loss)


def main(args):
    train_start = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    config = Config(args.config_file)
    model_path = f'{config.models_folder}/{args.model_name}-{train_start}.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = device == 'cuda'

    train_dataloader = create_dataloader(
        file_path=config.train_path,
        items='pairs',
        dataset=config.dataset,
        contexts_file=config.contexts_path,
        papers_file=config.papers_path,
        pad_token_id=config.pad_token_id,
        pad_author_id=config.pad_author_id,
        batch_size=config.train_queries_per_batch,
        mode='train',
        random_negs=config.random_negs
    )
    val_dataloader = create_dataloader(
        file_path=config.val_path, 
        items='triplets', 
        dataset=config.dataset,
        contexts_file=config.contexts_path,
        papers_file=config.papers_path,
        pad_token_id=config.pad_token_id,
        pad_author_id=config.pad_author_id,
        batch_size=config.triplets_batch_size,
        mode='val'
    )

    model = CitRecFunModel(
        embeddings_path=config.embeddings_path,
        rnn_num_layers=config.rnn_num_layers,
        rnn_hidden_size=config.rnn_hidden_size,
        author_embedding_dim=config.author_embedding_dim,
        num_author_embeddings=config.author_embedding_num,
        cnn_kernel_size=config.cnn_kernel_size,
        cnn_out_channels=config.cnn_out_channels,
        citations_dim=config.citations_dim,
        pad_author_id=config.pad_author_id,
        dual=args.dual,
        global_info=args.global_info,
        weight_scores=args.weighted_sum,
        device=device
    )
    if device == 'cuda':
        model = model.to(device)
    optimizer = Adam(model.parameters())

    best_validation_loss = float("Inf")
    for epoch in range(config.num_epochs):
        for i, (batch, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            preds = model(batch)
            batch_loss = triplet_loss(
                scores=preds, 
                margin=config.margin,
                device=device)
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            validation_loss = 0.
            for i, (batch, _) in enumerate(val_dataloader):
                preds = model(batch)
                batch_loss = triplet_loss(
                    scores=preds, 
                    margin=config.margin,
                    device=device)
                validation_loss += batch_loss

            logging.info(f'Avg validation loss in epoch {epoch + 1}: {validation_loss / (i + 1)}')

            if validation_loss < best_validation_loss:
                logging.info(f'Reached best validation loss in epoch {epoch + 1}. '
                             f'Saving model to: {model_path}')
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='JSON file with config parameters')
    parser.add_argument('model_name', help='name under which model will be stored')
    parser.add_argument('-dual', default=False, action='store_true', help='True if dual scoring is used, False otherwise')
    parser.add_argument('-global_info', default=False, action='store_true', help='True if global information is used for context representation, False otherwise')
    parser.add_argument('-weighted_sum', default=False, action='store_true', help='True if sum of scores is weighted, False otherwise')
    args = parser.parse_args()

    main(args)
