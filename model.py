import torch
from torch import nn

from utils import Config
from modules.attention import AttentionLayer
from modules.embedding import EmbeddingLayer, AuthorsCNN
from modules.rnn import LSTMLayer
from modules.mlp import MLPLayer


class CitRecFunModel(nn.Module):

    def __init__(
        self, 
        embeddings_path,
        rnn_num_layers,
        rnn_hidden_size,
        author_embedding_dim,
        num_author_embeddings,
        cnn_kernel_size,
        cnn_out_channels,
        pad_author_id,
        citations_dim,
        dual=False, 
        global_info=False, 
        weight_scores=True, 
        device='cpu'
    ):
        super().__init__()
        self.device = device
        self.dual = dual
        self.global_info = global_info
        self.weight_scores = weight_scores

        self.emb = EmbeddingLayer(embeddings_path)
        self.word_lstm = LSTMLayer(
            num_layers=rnn_num_layers, 
            hidden_size=rnn_hidden_size,
            input_size=self.emb.embedding_dim, 
            device=device
        ) 

        # self-attention
        self.context_attention = AttentionLayer(
            query_dim=2*self.word_lstm.hidden_size, 
            value_dim=2*self.word_lstm.hidden_size,
            device=device
        )
        # attention over cited article's text
        self.ref_paper_attention = AttentionLayer(
            query_dim=2*self.word_lstm.hidden_size,
            value_dim=2*self.word_lstm.hidden_size,
            device=device
        )
        # attention over citing article's text
        if self.global_info:
            self.citing_paper_attention = AttentionLayer(
                query_dim=2*self.word_lstm.hidden_size, 
                value_dim=2*self.word_lstm.hidden_size,
                device=device
            )
        self.semantic_score = torch.nn.CosineSimilarity(dim=-1)

        # dual version needs components for bibliographic score
        if self.dual:
            self.author_embedding = AuthorsCNN(
                num_author_embeddings=num_author_embeddings, 
                author_embedding_dim=author_embedding_dim,
                padding_idx=pad_author_id,
                kernel_sizes=cnn_kernel_size, 
                out_channels=cnn_out_channels
            )
            self.meta_score = MLPLayer(
                input_size=self.author_embedding.embedding_dim + citations_dim,
                output_size=1
            )
            if self.weight_scores:
                self.score_weights = torch.nn.Linear(
                    in_features=2*self.word_lstm.hidden_size, 
                    out_features=2
                )

    def _paper_attention_emb(self, paper, query, paper_lens, attn_layer):
        paper = self.emb(paper)                     # [B, max_sent_len, emb_dim]
        paper = self.word_lstm(paper, paper_lens)   # [B, max_sent_len, 2d]
        attn_scores = attn_layer(query, paper, paper_lens)
        return torch.bmm(attn_scores, paper), attn_scores      # [B, 1, 2d] 

    def _context_embedding(self, context, context_lens, tc_indices, citing_paper=None, citing_paper_lens=None):
        # word embeddings for context and pass it through word lstm
        batch_size = context.shape[0]
        context = self.emb(context)
        context = self.word_lstm(context, context_lens)

        # extract embeddings for TARGET token (store it in context_query)
        tc_indices = tc_indices.repeat(
            1, context.shape[-1]).view(batch_size, 1, context.shape[-1])
        context_query = torch.gather(context, 1, tc_indices)

        # attention over words in context for final context embeddings
        context_attn_scores = self.context_attention(context_query, context, context_lens)
        context = torch.bmm(context_attn_scores, context)   # (B, 1, 2d)
        if self.global_info:
            citing_paper, citing_paper_attn_scores = self._paper_attention_emb(
                citing_paper, 
                context, 
                citing_paper_lens,
                attn_layer=self.citing_paper_attention
            )
            context = context + citing_paper
        return context

    def _meta_score(self, authors, citations):
        authors = self.author_embedding(authors)
        meta_input = torch.cat((authors, citations), dim=-1)
        return self.meta_score(meta_input)

    def _total_score(self, context, sem_score, meta_score):
        scores = torch.cat((sem_score.unsqueeze(1), meta_score), dim=1)
        if self.weight_scores:
            score_weights = torch.softmax(torch.sigmoid(self.score_weights(context)), dim=-1)
            return torch.bmm(score_weights, scores).squeeze(1)
        else:
            return torch.sum(scores, dim=1)

    def forward(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)

        context = self._context_embedding(
            batch['contexts'], 
            batch['contexts_lens'], 
            batch['tc_indices'], 
            batch['citing_papers'], 
            batch['citing_papers_lens']
        )
        
        pos_paper, _ = self._paper_attention_emb(
            batch['pos_papers'], 
            context, 
            batch['pos_papers_lens'], 
            attn_layer=self.ref_paper_attention
        )
        neg_paper, _ = self._paper_attention_emb(
            batch['neg_papers'], 
            context, 
            batch['neg_papers_lens'], 
            attn_layer=self.ref_paper_attention
        )

        sem_score_pos = self.semantic_score(context, pos_paper)
        sem_score_neg = self.semantic_score(context, neg_paper)

        if self.dual:
            meta_score_pos = self._meta_score(batch['pos_authors'], batch['pos_citations'])
            meta_score_neg = self._meta_score(batch['neg_authors'], batch['neg_citations'])
            total_score_pos = self._total_score(context, sem_score_pos, meta_score_pos)
            total_score_neg = self._total_score(context, sem_score_neg, meta_score_neg)
            return sem_score_pos, sem_score_neg, meta_score_pos.squeeze(1), \
                meta_score_neg.squeeze(1), total_score_pos, total_score_neg
        else:
            return sem_score_pos, sem_score_neg
            

    def predict(self, batch): 
        for k in batch:
            batch[k] = batch[k].to(self.device)

        context = self._context_embedding(
            batch['contexts'], 
            batch['contexts_lens'], 
            batch['tc_indices'], 
            batch['citing_papers'], 
            batch['citing_papers_lens']
        )
        ref_paper, _ = self._paper_attention_emb(
            batch['ref_papers'], 
            context, 
            batch['ref_papers_lens'], 
            attn_layer=self.ref_paper_attention
        )

        sem_score_ref = self.semantic_score(context, ref_paper)
        if self.dual:
            meta_score_ref = self._meta_score(batch['ref_authors'], batch['ref_citations'])
            total_score_ref = self._total_score(context, sem_score_ref, meta_score_ref)
            return sem_score_ref, meta_score_ref.squeeze(1), total_score_ref
        else:
            return sem_score_ref

