import json
from math import log2

def year_from_id(paper_id):
    digits = int(paper_id[1:3])
    return 2000 + digits if digits < 60 else 1900 + digits

def mrr(ranks, k):
    rec_ranks = [1./r if r <= k else 0. for r in ranks]
    return sum(rec_ranks) / len(ranks)

def recall(ranks, k):
    return sum(r <= k for r in ranks) / len(ranks)

def ndcg(ranks, k):
    ndcg_per_query = sum(1 / log2(r + 1) for r in ranks if r <= k)
    return ndcg_per_query / len(ranks)


class Config(object):

    dataset = None
    train_path = "train_true_pairs_json"
    val_path = "val_triplets_json"
    contexts_path = "contexts_map_json"
    papers_path = "papers_map_json"
    embeddings_path = "ai2_embeddings.txt"
    models_folder = "path_to_folder_for_storing_models"
    predictions_folder = "path_to_folder_for_storing_predictions"

    rnn_num_layers = 1
    rnn_hidden_size = 100
    pad_token_id = 504339

    cnn_kernel_size = [1, 2]
    cnn_out_channels = [100, 100]

    author_embedding_dim = 50
    author_embedding_num = None
    pad_author_id = None
    citations_dim = None
    
    num_epochs = None
    random_negs = None
    train_queries_per_batch = None
    triplets_batch_size = None
    margin = None

    def __init__(self, config_file):
        config = json.load(open(config_file))
        self.__dict__.update(config)

    def dump(self, output_file):
        json.dump(self.__dict__, open(output_file, 'wt'))
        print(f'Stored CONFIG to: {output_file}')
