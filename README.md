# HTR-Seq2Seq

**HTR-Seq2Seq** is a word-level Handwritten Text Recognition (HTR) model that leverages a sequence-to-sequence architecture to recognize handwritten words from images.

## Acknowledgment

This repository is based on the original implementation from  
[omni-us/research-seq2seq-HTR](https://github.com/omni-us/research-seq2seq-HTR).

## How to Train

Example training command.
```
export CUDA_VISIBLE_DEVICES=0 && python3 main_torch_latest.py --dataset MIXED_SINGLE_LINE --run_id 36 --wandb 1
```

--dataset refers to the folder where the data is located.
The folder strucure should be as {baseDir}{split}/images/{dataset}/gt_RWTH.txt where gt_RWTH.txt is the ground truth file, which can be downloaded from the original repository mentioned above or by contacting us.
The {split} corresponds to the train, val, and test folders of the dataset.
--run_id is the experiment ID.
If you include --wandb 1, the training data will be logged to [Weights & Biases (wandb)](https://wandb.ai/).

## Our Usage and Contribution
We have modified the code by replacing:
1. the augmentation techniques used in the Seq2Seq model with those from  
[gayan68/HTR-CRNN-BestPractice](https://github.com/gayan68/HTR-CRNN-BestPractice).

2. the CER and WER calculation which is now based on the CER and WER in [gayan68/HTR-CRNN-BestPractice](https://github.com/gayan68/HTR-CRNN-BestPractice).
