# Incremental Sentence Processing Mechanisms in Autoregressive Transformer Language Models

This repo provides the code for the paper *Incremental Sentence Processing Mechanisms in Autoregressive
Transformer Language Models*. Using this code, you should be able to replicate all of the experiments in the paper, though some of the data used (in particular the Penn Treebank) is not publicly / freely available.

To replicate the results of the paper, please do the following:

1. Using conda, create an environment from the `environment.yml` file.
2. (Section 4.1, Figure 2): Run `python behavioral_evaluation.py`. For results with Gemma (Section D.1, Figure 7), run `python behavioral_evaluation.py --model_name google/gemma-2-2b`.
3. (Section 4.2, Figure 3, Table 2): Run . Then use the notebook `annotate_dashboard.ipynb` to annotate the features manually. Note that you'll have to add your own API key to do so.