# Incremental Sentence Processing Mechanisms in Autoregressive Transformer Language Models

This repo provides the code for the paper *Incremental Sentence Processing Mechanisms in Autoregressive
Transformer Language Models*. Using this code, you should be able to replicate all of the experiments in the paper, though some of the data used (in particular the Penn Treebank) is not publicly / freely available, and external data (that does not originate from this project) should be downloaded from its source.

## Replicate our Results

### Gather Results

To replicate the results of the paper, please do the following. Note that these commands, will, for the most part, just gather results, not plot them; see the next section for plotting.

1. Using conda, create an environment from the `environment.yml` file.
2. (Section 4.1, Figure 2): Run `python behavioral_evaluation.py`. For results with Gemma (Section D.1, Figure 7), run `python behavioral_evaluation.py --model_name google/gemma-2-2b`.
3. (Section 4.2, Figure 3, Figure 9-12, Table 2):
   1. Make sure that you've gotten the contents of the submodule `feature-circuits-gp` (as well as its submodule, `dictionary_learning`). To do this, use `git submodule update --init --recursive`
   2. Download the `pythia-70m-deduped` SAEs from [here](https://baulab.us/u/smarks/autoencoders/) into `feature-learning-gp/dictionaries`. For more details, see `dictionary_learning` / its README file.
   3. Run `feature-circuits-gp/scripts/get_circuit_garden_path.sh`. The annotation of each figure comes from the annotation file `feature-circuits-gp/annotations/pythia-70m-deduped.jsonl`. Figures 9-12 will be output in `feature-circuits-gp/circuits/figures`; however, note that our Figure 3 was constructed manually.
      1. The same script can be run for Gemma (just change the model in the script, but also consider changing the batch size). These SAEs will download automatically, but be forewarned that they are large.
   4. Use the notebook `annotate_dashboard.ipynb` to annotate the features manually. Note that you'll have to add your own Neuronpedia API key to do so.
   5. To compute faithfulness, please use `feature-circuits-gp/scripts/evaluate_circuit.sh`.  
4. (Section 4.3, Figure 4): Run `python causal_analysis.py`. For results with Gemma (Section D.1, Figure 8), run `python causal_analysis.py --model_name google/gemma-2-2b`. Note that this depends on the files `results/<model>/npz_features.csv` and `results/<model>/npz_features.csv`, which we crafted manually.
   1. (Appendix C, Figure 6) For large-scale results, first download [this file from the SAP Benchmark](https://github.com/caplabnyu/sapbenchmark/blob/main/Surprisals/data/items_ClassicGP.csv) to `data_csv`. Then run `python causal_analysis_largescale.py` (`pythia-70m-deduped` only).
5. (Section 5.1): Run `get_compare_activations.py` to get the values discussed in the paper (`pythia-70m-deduped` only).
6. (Section 5.2): We provide the probes trained on `pythia-70m-deduped` needed for the structural probing experiments at [this link](https://surfdrive.surf.nl/files/index.php/s/FksPx78imBQKKrn); you just need to download the probes and put them in a folder called `standalone_probes`. So, feel free to do that and skip steps 1 and 2.
   1. (Optional) First, train the structural probes on `pythia-70m-deduped`.
      1. For this, you will need (our fork of) `incremental_parse_probe`. Go to that fork, and create a new conda environment based on the `environment.yml` file there.
      2. Get a copy of Penn TreeBank, and put it in `incremental_parse_probe`.
      3. Generate the data splits by running `incremental_parse_probe/convert_splits_to_depparse.sh`. Then follow the steps in the *Preprocessing* section of that repo's README (involves using Stanford CoreNLP + Java and is rather complicated).
      4. Finally, run `incremental_parse_probe/inter_train_pythia_deduped.sh`.
      5. Copy the last checkpoint of each probe (in `incremental_parse_probe/experiment_checkpoints/eval/pythia-70m-deduped/StackActionProbe/layer_<layer>`) into `standalone_probes`. The probes should be named `embeddings.pt`, `layer0.pt`, ..., `layer5.pt`; note that `embeddings` corresponds to `layer_0` in the `experiment_checkpoints` folder.
   2. (Optional, Figure 14) Second, evaluate the probes by running `incremental_parse_probe/iter_eval_pythia_deduped.sh`. Copy the files in `incremental_parse_probe/results` to `results/pythia-70m-deduped/parse_probe/performance/`.
   3. (Figure 5, Appendix F, Figure 15): Run `parseprobe_behavior.py`.
   4. (Appendix F.4, Figure 16): Run `parseprobe_attribution.py`
7. (Section 6.1, Table 4): To evaluate on the reading comprehension questions, use the `readingcomp_evaluation.py` script. By default, this will evaluate Gemma 2 (2B) on `data_csv/garden_path_samelen_readingcomp.csv`. Use the `--model` argument to change the model (this takes a HuggingFace model identifier) or the `--data` argument to change the dataset to a different .csv from `data_csv/`.
8. (Section 6.2): To discover a feature circuit for reading comprehension, use `feature-circuits-gp/scripts/get_circuit_garden_path.sh`, but using the garden path sentences as the data. To compute feature overlaps, use `feature-circuits-gp/feature_overlap.py`. This script takes in the nodes from two circuits discovered using the circuit discovery scripts in `feature-circuits-gp/scripts/`. Only the nodes are needed for this analysis.

### Create Plots

Once you've run the corresponding line above, you can create each figure by running the following files within `plotting/`:

- Figure 1, 9, 13: Manually created
- Figure 2, 7: `plotting/behavioral-subplots-difference.py`
- Figure 4, 8: `plotting/causal-subplots-difference.py`
- Figure 5, 15: `plotting/parse-probe-behavior.py`
- Figure 6: `plotting/causal-subplots-largescale-difference.py`
- Figure 14: `plotting/parse-probe-performance.py`
- Figure 16: `plotting/parse-probe-overlap.py`

## Data

As part of this project, we created the following data files:

- `data_csv/gp_same_len.csv`: An edit of Arehalli et al.'s (2022) dataset containing both ambiguous and unambiguous sentences, all of the same length. Note that our unambiguous sentences are unambiguous because of the verb used not, e.g. because of an added comma.
- `data_csv/garden_path_readingcomp.csv`: An adaptation of the above dataset containing complete garden path sentences and follow-up questions.
- `data_csv/garden_path_samelen_readingcomp.csv`: A version of the above dataset containing garden path sentences that each contain the same number of words. This version of the dataset enables us to analyze which sparse features are most influential at specific token positions.

## Citation

Coming soon!

## License

We release our materials under an MIT license.
