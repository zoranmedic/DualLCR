# Improved Local Citation Recommendation Based on Context Enhanced with Global Information

This repository is the official implementation of [Improved Local Citation Recommendation Based on Context Enhanced with Global Information](https://www.aclweb.org/anthology/2020.sdp-1.11). 

Repository includes code for training and evaluating the proposed model, together with links to data and resources downloads. If you encounter any problems or errors feel free to raise an issue or send an email to <zoran.medic@fer.hr>.

## Data and Resources

We provide links for downloading preprocessed dataset instances as well as training, validation, and test splits. Both ACL-ARC and RefSeer dataset files are compressed and available on the following links:
* [ACL-ARC](https://drive.google.com/file/d/1i-0cmwTM7rBL937PoPBK3mFLvGBusJLS/view?usp=sharing)
* [RefSeer](https://drive.google.com/file/d/13ueqHTn2863EJRVhsK2Itxc2OovlBd0f/view?usp=sharing)

For running the experiments, you will need the file with pretrained word embeddings. While the code should run with whatever embeddings you have (either pretrained or not), you should make sure to include the following tokens in the vocabulary: `'TARGETCIT', 'OTHERCIT', 'PAD', 'UNK'`. 
[Here](https://drive.google.com/file/d/1iiIu1Rz9iGPXs4La5_57CcJVQ8dxEc5j/view?usp=sharing) you can download the pretrained embeddings that we used in our experiments. The embeddings are from the paper [''Content-Based Citation Recommendation"](https://www.aclweb.org/anthology/N18-1022/) by Bhagavatula et al. (2018). In the file that we used, we included the above mentioned tokens in the vocabulary (randomly initialized) and used that in our experiments.

## Requirements

Model is built using PyTorch. For setting up the environment for running the code, make sure you run the following commands:

```
git clone git@github.com:zoranmedic/DualLCR.git

conda env create -f environment.yml

conda activate testenv

python -m spacy download en_core_web_sm

```

## Training

To train the 'DualEnh-ws' variant of the model described in the paper, run this command:

```train
python train.py <path_to_config_json> model_name -dual -global_info -weighted_sum
```

Make sure you fill out all the relevant fields in the config JSON file.


## Evaluation

To generate predictions on test files with 'DualEnh-ws' variant of the model, run the following command:

```eval
python predict.py <path_to_config_json> <path_to_saved_model> -dual -global_info -weighted_sum <path_to_input_file>
```


## Citation

Please cite our work as:
```
@inproceedings{medic-snajder-2020-improved,
    title = "Improved {L}ocal {C}itation {R}ecommendation {B}ased on {C}ontext {E}nhanced with {G}lobal {I}nformation",
    author = "Medi{\'c}, Zoran and \v{S}najder, Jan",
    booktitle = "Proceedings of the First Workshop on Scholarly Document Processing",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.sdp-1.11",
    pages = "97--103"
}
```
