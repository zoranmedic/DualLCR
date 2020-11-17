import argparse
import json
import spacy

nlp = spacy.load('en_core_web_sm')


def tokenize(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc]


def tokenize_and_index(text, word2index, vocabulary, split_sentences=False):
    tokenids = []
    if split_sentences:
        doc = nlp(text)
        for sent in doc.sents:
            sent_tokens=[]
            for token in sent:
                t = token.text.lower()
                if t in vocabulary:
                    sent_tokens.append(word2index[t])
                else:
                    sent_tokens.append(word2index['UNK'])
            tokenids.append(sent_tokens)
    else:
        tokens = tokenize(text)
        for token in tokens:
            if token in vocabulary:
                tokenids.append(word2index[token])
            else:
                tokenids.append(word2index['UNK'])
    return tokenids


def main(args):
    word2index = {}
    c = 0
    with open(args.embeddings_file) as f:
        next(f)  # skip the first line
        for line in f:
            word2index[line.split()[0]] = c
            c += 1
    vocabulary = set(list(word2index.keys()))

    data = json.load(open(args.input_file))
    print('Total:', str(len(data)))
    for i, d in enumerate(data):
        if i % 10 == 0:
            print(i)
        if args.papers:
            data[d]['preprocessed_abstract'] = tokenize_and_index(data[d]['abstract'], word2index, vocabulary, split_sentences=True)
            data[d]['preprocessed_title'] = tokenize_and_index(data[d]['title'], word2index, vocabulary)
        else:
            data[d]['preprocessed'] = tokenize_and_index(data[d]['masked_text'], word2index, vocabulary)
            data[d]['tc_index'] = data[d]['preprocessed'].index(word2index['targetcit'])
    
    json.dump(data, open(args.input_file, 'wt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='JSON file with ids as keys and dictionary of fields as values')
    parser.add_argument('embeddings_file', help='txt file with embeddings for tokens, used for determining vocabulary')
    parser.add_argument('-papers', default=False, action='store_true', help='declares that input file contains papers, if not given contexts are assumed')
    args = parser.parse_args()

    main(args)
