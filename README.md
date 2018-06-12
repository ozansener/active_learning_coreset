# Active Learning via Core-Sets
Source code for ICLR 2018 Paper: Active Learning for Convolutional Neural Networks: A Core-Set Approach

## Main Organization of the Code	

- additional_baselines:
  - This folder includes baselines as well as pytorch implementation of the CIFAR-10 VGG network training code.
- coreset
  - This folder includes the discrete optimization code which given feature emeddings, solves for core-sets. Its output chosen ids which is further used by learning code.
- tf_base
  - Tensorflow implementation which trains and test the model. It uses the output of the coreset which includes chosen data points.

  ## Reference

If you find the code useful, please cite the following papers:

Active Learning for Convolutional Neural Networks: A Core-Set Approach. O. Sener, S. Savarese. International Conference on Learning Representations (ICLR), 2018. ()

    @inproceedings{sener2018active,
        title={Active Learning for Convolutional Neural Networks: A Core-Set Approach},
        author={Ozan Sener and Silvio Savarese},
        booktitle={International Conference on Learning Representations},
        year={2018},
        url={https://openreview.net/forum?id=H1aIuk-RW},
    }