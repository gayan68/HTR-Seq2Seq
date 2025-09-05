# HTR-Seq2Seq

**HTR-Seq2Seq** is a word-level Handwritten Text Recognition (HTR) model that leverages a sequence-to-sequence architecture to recognize handwritten words from images.

## Acknowledgment

This repository is based on the original implementation from  
[omni-us/research-seq2seq-HTR](https://github.com/omni-us/research-seq2seq-HTR).  

We have modified the code by replacing:
1. the augmentation techniques used in the Seq2Seq model with those from  
[gayan68/HTR-CRNN-BestPractice](https://github.com/gayan68/HTR-CRNN-BestPractice).

2. the CER and WER calculation which is now based on the CER and WER in [gayan68/HTR-CRNN-BestPractice](https://github.com/gayan68/HTR-CRNN-BestPractice).
