import argparse
import json
import re
import spacy

nlp = spacy.load('en_core_web_sm')

# Regular expressions for masking citations in input texts
author_regex = "[A-Z][a-z]+((\s|-)[A-Z][a-z]+)*"
year_regex = "[0-9]{4}[a-z]?"
cit_brackets_regex = "\[(\d+\s?,\s?)*\d+\]"
refseer_tc_regex = "=-=(.*)-=-"
mul_other_regex = '(OTHERCIT(;|,)\s)+OTHERCIT'

citation_regex_defs = [
    # citep
    [author_regex + ",", year_regex], # single author + year
    [author_regex, "&", author_regex + ",?", year_regex], # two authors + year
    [author_regex + ",", author_regex + ",", "&", author_regex + ",?", year_regex], # three authors w/ & + year
    [author_regex + ",", author_regex + ",", "and", author_regex + ",?", year_regex], # three authors w/ and + year
    [author_regex, "and", author_regex + ",?", year_regex], # two authors + year
    [author_regex, "et al\.?" + ",?", year_regex], # multiple authors + year
    # citet
    [author_regex, "\(" + year_regex + "\)"], # single author
    [author_regex, "&", author_regex, "\(" + year_regex + "\)"], # two authors
    [author_regex + ",", author_regex + ",", "&", author_regex, "\(" + year_regex + "\)"], # three authors
    [author_regex + ",", author_regex + ",", "and", author_regex, "\(" + year_regex + "\)"], # three authors
    [author_regex, "and", author_regex, "\(" + year_regex + "\)"], # two authors
    [author_regex, "et al\.", "\(" + year_regex + "\)"] # multiple authors
]
citation_regexes = ['\s'.join(l) for l in citation_regex_defs]


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


def mask_citations(text, dataset='other'):
    if dataset == 'acl':
        text = text[:600] + ' TARGETCIT ' + text[-600:]  # assumes longer contexts in ACL
    elif dataset == 'refseer':
        text = re.sub(refseer_tc_regex, 'TARGETCIT', text)
    
    citation_regex = '|'.join(citation_regexes)
    if dataset == 'refseer':
        citation_regex += '|' + cit_brackets_regex
    
    masked = re.sub(citation_regex, 'OTHERCIT', text)
    masked = re.sub(mul_other_regex, 'OTHERCIT', masked)
    masked = re.sub(' +', ' ', masked)
    return masked


def main(args):
    word2index = {}
    c = 0
    # Generate vocabulary from tokens in the embeddings file
    with open(args.embeddings_file) as f:
        next(f)  # skip the first line
        for line in f:
            word2index[line.split()[0]] = c
            c += 1
    vocabulary = set(list(word2index.keys()))

    data = json.load(open(args.input_file))
    print(f"Total data instances to preprocess: {str(len(data))}")

    for i in data:
        if args.input_type == 'articles':
            data[i]['preprocessed_abstract'] = tokenize_and_index(data[i]['abstract'], word2index, vocabulary, split_sentences=True)
            data[i]['preprocessed_title'] = tokenize_and_index(data[i]['title'], word2index, vocabulary)
        else: # preprocess contexts
            if args.mask_contexts:
                data[i]['masked_text'] = mask_citations(data[i]['citation_context'], dataset=args.dataset)
            data[i]['preprocessed'] = tokenize_and_index(data[i]['masked_text'], word2index, vocabulary)
            data[i]['tc_index'] = data[i]['preprocessed'].index(word2index['targetcit'])
    
    json.dump(data, open(args.input_file, 'wt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='JSON file with ids as keys and dictionary of fields as values')
    parser.add_argument('embeddings_file', help='txt file with embeddings for tokens, used for determining vocabulary')
    parser.add_argument('-input_type', default='articles', type=str, help='what type of text is in the input: [articles, contexts]')
    parser.add_argument('--mask_contexts', default=False, action='store_true', help='whether to include the step of masking the texts in the context')
    parser.add_argument('--dataset', default='other', type=str, help='type of dataset (needed if masking contexts): [acl, refseer, other]')
    args = parser.parse_args()

    main(args)
