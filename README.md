# Attention Pruning

This repo hosts code for [Data-Informed Global Sparseness in Attention Mechanisms for Deep Neural Networks](https://arxiv.org/abs/2012.02030) and borrows starter code from [fairseq](https://github.com/pytorch/fairseq), [huggingface](https://github.com/huggingface/transformers), and [transformer-xl](https://github.com/kimiyoung/transformer-xl).


## Results


### Performance
<img src="./ap_figure.png" width="600px"></img>

See paper for details, comparison with entmax, and ood results.

### Computational Efficiency
Results on the SQuAD question answering task:


| Percentage  | Exact/F1 scores | Time(s) | GPU Memory(GB) |
| ----------- | --------------- | ------- | ---------------|
| 0           | 81.02/88.63     | 95.41   |6.85            |
| 90          | 79.62/87.32     | 86.44   |5.00            |
