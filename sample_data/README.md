Sample JSON files that can be used as input to `dataset/preprocess.py` for obtaining lists of indices representing tokens in both contexts and articles. 
Additionally, contexts are given without `masked_text` field, so the text in them needs to be preprocessed with `mask_citations` function, which replaces citations with placeholders.
Both contexts and articles in sample files are from the ACL-ARC dataset.

Example for running the preprocessing of contexts file (run from project's root folder):
```
python datasets/preprocess.py sample_data/sample_contexts.json ~/resources/ai2_embeddings.txt -input_type contexts --mask_contexts --dataset acl
```

Example for running the preprocessing of articles file (run from project's root folder):
```
python datasets/preprocess.py sample_data/sample_articles.json ~/resources/ai2_embeddings.txt -input_type articles
```
