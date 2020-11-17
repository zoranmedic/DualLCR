import argparse
import json

from utils import mrr, recall, ndcg


def main(args):
    ranks = []
    for f in args.input_files:
        preds = json.load(open(f))
        for cid in preds:
            sorted_preds = [  # paper ids sorted by recommendation score
                i[0] for i in sorted(preds[cid], key=lambda x: x[1], reverse=True)
            ]
            rank = sorted_preds.index(cid.split('_')[1]) + 1  # rank of correct recommendation
            ranks.append(rank)

    print(f'Recall@k: {recall(ranks, k=10):.5f}')
    print(f'MRR: {mrr(ranks, k=10):.5f}')
    print(f'NDCG: {ndcg(ranks, k=10):.5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs="+", help='JSON files containing recommendation scores')
    args = parser.parse_args()

    main(args)
