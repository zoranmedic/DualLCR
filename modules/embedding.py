import gensim
import torch
import torch.nn.functional as F
from torch import nn


class EmbeddingLayer(nn.Module):

    def __init__(self, embeddings_path):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(self._load_embeddings(embeddings_path)), 
            freeze=False
        )
        self.embedding_dim = self.emb.embedding_dim

    def _load_embeddings(self, embeddings_path):
        return gensim.models.KeyedVectors.load_word2vec_format(embeddings_path).vectors

    def forward(self, x):
        return self.emb(x)


class AuthorsCNN(nn.Module):

    def __init__(self, num_author_embeddings, author_embedding_dim, padding_idx, kernel_sizes, out_channels):
        super().__init__()
        self.embedding_dim = sum(out_channels)
        self.author_emb = nn.Embedding(
            num_embeddings=num_author_embeddings, 
            embedding_dim=author_embedding_dim, 
            padding_idx=padding_idx
        )
        self.author_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=author_embedding_dim, 
                out_channels=channels, 
                kernel_size=size
            ) for size, channels in zip(kernel_sizes, out_channels)
        ])

    def _conv(self, conv, x):
        x = conv(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.permute((0, 2, 1))
        return x

    def forward(self, x):
        out = self.author_emb(x)
        out = out.permute(0, 2, 1)
        out = [self._conv(conv, out) for conv in self.author_convs]
        return F.relu(torch.cat(out, dim=-1))