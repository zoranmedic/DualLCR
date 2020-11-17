import json
import random
from collections import namedtuple

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import Config, year_from_id


Context = namedtuple('Context', ['context_tokens', 'tc_index', 'paper_tokens', 'authors', 'context_id'])
Paper = namedtuple('Paper', ['paper_tokens', 'authors', 'paper_id', 'citations'])
Triplet = namedtuple('Triplet', ['context', 'pos_paper', 'neg_paper'])
Pair = namedtuple('Pair', ['context', 'paper'])


class AbstractContextsPapersDataset(Dataset):

    def __init__(self, dataset, contexts_file, papers_file, pad_author_id, max_paper_tokens=200, max_authors=5):
        super().__init__()
        self.dataset = dataset
        self.contexts = json.load(open(contexts_file))
        self.papers = json.load(open(papers_file))
        self.max_paper_tokens = max_paper_tokens
        self.max_authors = max_authors
        self.pad_author_id = pad_author_id
        self.paper_ids = list(self.papers.keys())
        self.paper_ids_set = set(self.paper_ids)

    def context_tokens(self, context_id):
        return torch.tensor(self.contexts[context_id]['preprocessed'])

    def paper_tokens(self, paper_id):
        title_tokens = self.papers[paper_id]['preprocessed_title']
        abstract_sents = self.papers[paper_id]['preprocessed_abstract']
        abstract_tokens = [token for sentence in abstract_sents for token in sentence]
        paper = title_tokens + abstract_tokens
        return torch.tensor(paper[:self.max_paper_tokens])

    def authors_tokens(self, paper_id):
        tokens = self.papers[paper_id]['authors_ids'][:self.max_authors]
        if len(tokens) < 2:
            tokens += [self.pad_author_id for _ in range(2 - len(tokens))]
        return torch.tensor(tokens)

    def paper_citations(self, paper_id, citing_id, last_years=4):
        if self.dataset == 'refseer':
            return torch.tensor([self.papers[paper_id]['total_citations']]).float()
        else:
            paper_year = year_from_id(paper_id)
            citing_year = year_from_id(citing_id)
            
            total_citations = 0
            citation_years = [int(i) for i in self.papers[paper_id]['citations_per_year'].keys()]
            min_year = min(citation_years) if citation_years else citing_year+1
            for year in range(min_year, citing_year+1):
                total_citations += self.papers[paper_id]['citations_per_year'].get(str(year), 0)

            citations_per_year = []
            for year in range(citing_year, citing_year-last_years, -1):
                if year < paper_year:
                    num_citations = -1
                else:
                    num_citations = self.papers[paper_id]['citations_per_year'].get(str(year), 0)
                citations_per_year.append(num_citations)
            return torch.tensor(citations_per_year + [total_citations]).float()

    def get_negative_examples(self, context_id, n=100):
        citing, cited = context_id.split('_')[:2]  # citing and cited paper ids
        random_papers = set()
        while len(random_papers) < n:
            paper_ids = random.sample(self.paper_ids, n)
            for pid in paper_ids:
                if pid in [citing, cited]:
                    continue
                if self.dataset == 'acl':
                    if year_from_id(pid) > year_from_id(citing):  # skip new papers
                        continue
                random_papers.add(pid)
        return list(random_papers)[:n]


class PairsDataset(AbstractContextsPapersDataset):

    def __init__(self, dataset, contexts_file, papers_file, pad_author_id, pairs_path):
        super().__init__(dataset, contexts_file, papers_file, pad_author_id)
        if pairs_path.endswith('.json'):
            self.pairs = json.load(open(pairs_path))
        else:
            self.pairs = []
            with open(pairs_path, 'rt') as f:
                for line in f:
                    elems = line.strip().split(',')
                    for p in elems[1:]:
                        self.pairs.append({
                            'context_id': elems[0],
                            'paper_id': p
                        })
        self.dataset_size = len(self.pairs)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        pair = self.pairs[index]
        context_id, ref_id = pair['context_id'], pair['paper_id']
        citing_id = context_id.split('_')[0]

        context = Context(
            context_tokens=self.context_tokens(context_id), 
            tc_index=self.contexts[context_id]['tc_index'], 
            paper_tokens=self.paper_tokens(citing_id), 
            authors=self.authors_tokens(citing_id), 
            context_id=context_id
        )
        ref_paper = Paper(
            paper_tokens=self.paper_tokens(ref_id), 
            authors=self.authors_tokens(ref_id), 
            paper_id=ref_id, 
            citations=self.paper_citations(ref_id, citing_id)
        )
        return Pair(context=context, paper=ref_paper)


class TripletsDataset(AbstractContextsPapersDataset):

    def __init__(self, dataset, contexts_file, papers_file, pad_author_id, triplets_path):
        super().__init__(dataset, contexts_file, papers_file, pad_author_id)
        self.triplets = json.load(open(triplets_path))
        self.dataset_size = len(self.triplets)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        triplet = self.triplets[index]
        context_id, citing_id = triplet['context_id'], triplet['context_id'].split('_')[0]
        pos_id, neg_id = triplet['true_ref'], triplet['neg_ref']

        context = Context(
            context_tokens=self.context_tokens(context_id), 
            tc_index=self.contexts[context_id]['tc_index'], 
            paper_tokens=self.paper_tokens(citing_id), 
            authors=self.authors_tokens(citing_id), 
            context_id=context_id
        )
        pos_paper = Paper(
            paper_tokens=self.paper_tokens(pos_id), 
            authors=self.authors_tokens(pos_id), 
            paper_id=pos_id, 
            citations=self.paper_citations(pos_id, citing_id)
        )
        neg_paper = Paper(
            paper_tokens=self.paper_tokens(neg_id), 
            authors=self.authors_tokens(neg_id), 
            paper_id=neg_id, 
            citations=self.paper_citations(neg_id, citing_id)
        )
        return Triplet(context=context, pos_paper=pos_paper, neg_paper=neg_paper)

