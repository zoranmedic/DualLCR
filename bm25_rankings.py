import argparse
import json

from rank_bm25 import BM25Okapi

from datasets.preprocess import tokenize
from utils import year_from_id


def main(args):
    contexts_per_year = json.load(open(args.contexts_per_year))
    contexts = json.load(open(args.contexts_file))
    context_hardnegatives = {}

    # preprocess all papers
    papers = json.load(open(args.papers_file))
    print('Tokenizing papers...')
    tokenized_papers = {
        k: tokenize(papers[k]['title'] + ' ' + papers[k]['abstract'])
        for k in papers
    }
    print(f'Tokenized {len(tokenized_papers)} papers')
    
    # go through all years
    for year in contexts_per_year:
        # keep only papers published up to and including current year
        year_tokenized_papers, pids = [], []
        for pid in tokenized_papers:
            if year_from_id(pid) <= int(year):
                year_tokenized_papers.append(tokenized_papers[pid])
                pids.append(pid)

        # preprocess contexts from current year
        tokenized_contexts = []
        print('Tokenizing contexts...')
        for cid in contexts_per_year[year]:
            c = contexts[cid]
            citing_title_abstract = papers[c['citing_paper_id']]['title'] + ' ' + papers[c['citing_paper_id']]['abstract']
            context = c['masked_text']
            tokenized_contexts.append((tokenize(citing_title_abstract + ' ' + context), cid))
        print(f'Tokenized {len(tokenized_contexts)} contexts')

        # fit BM25 on papers for current year
        model = BM25Okapi(year_tokenized_papers)
        
        # get BM25 scores for contexts from current year
        for context, cid in tokenized_contexts:
            citing, cited = cid.split('_')[:2]
            scores = model.get_scores(context)
            
            # sort pids by scores and keep top k + 1
            sorted_pids = [pid for pid, _ in sorted(list(zip(pids, scores)), key=lambda x: x[1], reverse=True)][:args.k+1]
            
            # remove citing id if in candidates
            if citing in sorted_pids:
                sorted_pids.remove(citing)
            sorted_pids = sorted_pids[:args.k]
            
            # add cited id if not in candidates
            if cited not in sorted_pids:
                sorted_pids[-1] = cited
            
            context_hardnegatives[cid] = sorted_pids

    json.dump(context_hardnegatives, open(args.output_file, 'wt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('contexts_file', help='JSON file with context texts')
    parser.add_argument('contexts_per_year', help='JSON file with years as keys and list of context ids as values')
    parser.add_argument('papers_file', help='JSON file with paper title and abstracts')
    parser.add_argument('output_file', help='JSON file in which dictionary of context ids as keys and list of paper ids as values is written')
    parser.add_argument('-k', type=int, default=2000, help='number of candidates to produce for each context')
    args = parser.parse_args()

    main(args)
