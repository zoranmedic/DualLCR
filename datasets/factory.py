from torch.utils.data import DataLoader

from .dataset import TripletsDataset, PairsDataset
from .collate import collate_pairs_train, collate_pairs_predict, collate_triplets

def create_dataloader(
    file_path, 
    items,
    dataset,
    contexts_file,
    papers_file,
    pad_token_id, 
    pad_author_id, 
    batch_size, 
    mode, 
    random_negs=None
    ):

    # create Dataset object of either pairs or triplets
    if items == 'triplets':
        dataset = TripletsDataset(dataset, contexts_file, papers_file, pad_author_id, file_path)
    elif items == 'pairs':
        dataset = PairsDataset(dataset, contexts_file, papers_file, pad_author_id, file_path)

    # create collate function specific for train/val/test dataset
    shuffle = True
    if mode == 'train':
        collate_fn = lambda x: collate_pairs_train(
            x, pad_token_id, pad_author_id, dataset, num_negs=random_negs
        )
    elif mode == 'val':
        collate_fn = lambda x: collate_triplets(
            x, pad_token_id, pad_author_id
        )
    elif mode == 'test' or mode == 'val':
        collate_fn = lambda x: collate_pairs_predict(
            x, pad_token_id, pad_author_id
        )
        shuffle = False

    # create Dataloader object with specified Dataset and collate function
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    
    return dataloader
