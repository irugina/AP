# AP

Code for the paper "Data-Informed Global Sparseness in Attention Mechanisms for Deep Neural Networks"

Initial codebases:

<ul>
<li>https://github.com/pytorch/fairseq</li>
<li>https://github.com/huggingface/transformers</li>
<li>https://github.com/kimiyoung/transformer-xl</li>
</ul>


## Transformer-XL

We only experiment with pruning wikitext-103 base architecture. To replicate our results with either this or other datasets, you might need to modify some or all of:

<ul>
<li>the constants defined in the first few lines of compute_mask.py </li>
<li>last line of compute_mask which specifies where to save the average attention pattern.</li>
</ul>	


Note that train.py takes an argument ```fn``` to specify path to average attention pattern.
