import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import Triplet, Paper


def pad_and_lens(sequence, padding_value):
    seq_lens = torch.tensor([i.size(0) for i in sequence])
    sequence = pad_sequence(sequence, batch_first=True, padding_value=padding_value)
    return sequence, seq_lens


def pad_and_indices(sequence, padding_value):
    padded, _ = pad_and_lens(sequence, padding_value)
    mask = (torch.ones(padded.shape) * padding_value).long()
    indices = padded != mask
    return padded, indices


def collate_pairs_predict(batch, pad_token_id, pad_author_id):
    contexts, tc_indices, citing_papers, citing_authors, ref_papers, ref_authors, \
        ref_citations, context_ids, ref_ids = zip(*[(
        pair.context.context_tokens, 
        pair.context.tc_index, 
        pair.context.paper_tokens,
        pair.context.authors,
        pair.paper.paper_tokens,  
        pair.paper.authors, 
        pair.paper.citations,
        pair.context.context_id,
        pair.paper.paper_id) for pair in batch
    ])
    
    contexts, contexts_lens = pad_and_lens(contexts, pad_token_id)
    tc_indices = torch.tensor(tc_indices)
    citing_papers, citing_papers_lens = pad_and_lens(citing_papers, pad_token_id)

    ref_papers, ref_papers_lens = pad_and_lens(ref_papers, pad_token_id)
    ref_authors, _ = pad_and_indices(ref_authors, pad_author_id)
    ref_citations = torch.cat(ref_citations).view((len(batch), 1, -1)).float()
    
    return {
        'contexts': contexts, 
        'contexts_lens': contexts_lens,
        'tc_indices': tc_indices, 
        'citing_papers': citing_papers, 
        'citing_papers_lens': citing_papers_lens,
        'ref_papers': ref_papers, 
        'ref_papers_lens': ref_papers_lens,
        'ref_authors': ref_authors,
        'ref_citations': ref_citations
    }, {
        'context_ids': context_ids,
        'ref_ids': ref_ids
    }


def collate_triplets(batch, pad_token_id, pad_author_id):
    """Collates list of triplets into tensors ready for input to model.

    Args:
        batch (list): List of triplet items. Each item is represented 
            as tuple: (dict, (context_id, pos_id, ref_id)).

    Returns:
        bool: The return value. True for success, False otherwise.

    """
    contexts, tc_indices, citing_papers, pos_papers, pos_authors, \
        neg_papers, neg_authors, context_ids, pos_ids, neg_ids, \
            pos_citations, neg_citations = zip(*[(
        triplet.context.context_tokens, 
        triplet.context.tc_index, 
        triplet.context.paper_tokens, 
        triplet.pos_paper.paper_tokens, 
        triplet.pos_paper.authors,
        triplet.neg_paper.paper_tokens,
        triplet.neg_paper.authors,
        triplet.context.context_id,
        triplet.pos_paper.paper_id, 
        triplet.neg_paper.paper_id,
        triplet.pos_paper.citations,
        triplet.neg_paper.citations) for triplet in batch
    ])

    contexts, contexts_lens = pad_and_lens(contexts, pad_token_id) 
    tc_indices = torch.tensor(tc_indices)
    citing_papers, citing_papers_lens = pad_and_lens(citing_papers, pad_token_id)

    pos_papers, pos_papers_lens = pad_and_lens(pos_papers, pad_token_id)
    neg_papers, neg_papers_lens = pad_and_lens(neg_papers, pad_token_id)
    pos_authors, _ = pad_and_indices(pos_authors, pad_author_id)
    neg_authors, _ = pad_and_indices(neg_authors, pad_author_id)

    pos_citations = torch.cat(pos_citations).view((len(batch), 1, -1)).float()
    neg_citations = torch.cat(neg_citations).view((len(batch), 1, -1)).float()

    return {
        'contexts': contexts,
        'contexts_lens': contexts_lens,
        'tc_indices': tc_indices, 
        'citing_papers': citing_papers, 
        'citing_papers_lens': citing_papers_lens,
        'pos_papers': pos_papers,
        'pos_papers_lens': pos_papers_lens,
        'neg_papers': neg_papers,
        'neg_papers_lens': neg_papers_lens,
        'pos_authors': pos_authors,
        'neg_authors': neg_authors,
        'pos_citations': pos_citations,
        'neg_citations': neg_citations
    }, {
        'context_ids': context_ids, 
        'pos_ids': pos_ids, 
        'neg_ids': neg_ids
    }


def collate_pairs_train(batch, pad_token_id, pad_author_id, dataset=None, num_negs=100):
    negatives_batch = []
    for pair in batch:  # random sampling for triplets
        negative_papers = dataset.get_negative_examples(
            context_id=pair.context.context_id, n=num_negs
        )
        citing_id = pair.context.context_id.split('_')[0]
        for paper_id in negative_papers:
            neg_paper = Paper(
                paper_tokens=dataset.paper_tokens(paper_id), 
                authors=dataset.authors_tokens(paper_id), 
                paper_id=paper_id, 
                citations=dataset.paper_citations(paper_id, citing_id)
            )
            triplet = Triplet(
                context=pair.context,
                pos_paper=pair.paper,
                neg_paper=neg_paper
            )
            negatives_batch.append(triplet)
    return collate_triplets(negatives_batch, pad_token_id, pad_author_id)
