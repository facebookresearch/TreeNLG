## Constrained Decoding for Neural NLG from Compositional Representations in Task-Oriented Dialogue

Code and dataset supporting the paper:

Anusha Balakrishnan, Jinfeng Rao, Kartikeya Upasani, Michael White and Rajen Subba. [Constrained Decoding for Neural NLG from Compositional Representations in Task-Oriented Dialogue](https://www.aclweb.org/anthology/P19-1080/).

If you find this code or dataset useful in your research, please consider citing our paper.

## Reference

```
@inproceedings{balakrishnan-etal-2019-constrained,
  title = "Constrained Decoding for Neural {NLG} from Compositional Representations in Task-Oriented Dialogue",
  author = "Balakrishnan, Anusha  and
    Rao, Jinfeng  and
    Upasani, Kartikeya  and
    White, Michael  and
    Subba, Rajen",
  booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
  month = jul,
  year = "2019",
  address = "Florence, Italy",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/P19-1080",
  doi = "10.18653/v1/P19-1080",
  pages = "831--844"
}
```

## Data

In addition to the **weather** and enriched **[E2E challenge](https://github.com/tuetschek/e2e-dataset)** dataset from our paper, we released another **weather_challenge** dataset, which contains harder weather scenarios in train/val/test files.
Each response was collected by providing annotators, who are native English speakers, with a *user query*, and a *compositional meaning representation* (with discourse relations and dialog acts). All of these are made available in our dataset. See our linked paper for more details.

#### Data Statistics

Dataset           | Train | Val  | Test | Disc_Test
-------           | ----- | ---  | ---- | ---------
Weather           | 25390 | 3078 | 3121 | 454
Weather_Challenge | 32684 | 3397 | 3382 | -
E2E               | 42061 | 4672 | 4693 | 230

`Disc_Test` is a more challenging subset of our test set that contains discourse relations, which is also the subset we report results in `Disc` column in Table 7 in our paper. Note that there are some minor differences of data statistics to our paper, please use the statistics above.

Note: There are some responses in `Weather` dataset which are not provided a user query (141/17/18/4 for train/val/test/disc_test, respectively).  We simply use a "placeholder" token for those missing user queries.

## Code

Computing tree accuracy:

```
python scripts/compute_tree_acc.py -tsv example/seq2seq_out.tsv
```

This should give you 94.65% tree accuracy. Output file should be tab-separated with columns `id, input, pred`.

### Constrained Decoding

[fairseq](https://github.com/pytorch/fairseq) should be installed at the very begining, refering to [Requirements and Installation of Fairseq](https://github.com/pytorch/fairseq#requirements-and-installation). The code is tested on commit `6234e29` of [fairseq](https://github.com/pytorch/fairseq). Now, get started with the following bash scripts.

```bash
bash scripts/prepare.weather.sh # preparing data
bash scripts/preprocess.weather.sh
bash scripts/train.weather.lstm.sh # training a simple model
bash scripts/generate.weather.lstm.sh # generating with constrainted decoding
```

### Results

We noticed that slightly higher numbers can be obtained by tuning hyper-parameters compared to the numbers we reported in our paper. Therefore, we update all the automatic numbers (BLEU and tree accuracy) here and please use numbers below when citing our results. For tree accuracy, we report the number on the whole test set, as well as on two disjoint subsets: **no-discourse** subset that contains examples without any discourse act; **discourse** subset contains example with 1+ discourse acts.

Note: The BLEU score is calculated on just the output text, without any of the tree information. We use the BLEU evaluation script provided for the E2E challenge [here](https://github.com/tuetschek/e2e-metrics).

##### Weather Dataset

Dataset    | BLEU  | TreeAcc(whole) | TreeAcc(no-discourse) | TreeAcc(discourse)
-------    | ----  | -------------- | --------------------- | ------------------
S2S-Tree   | 76.12 | 94.00          | 96.66                 | 86.59
S2S-Constr | 76.60 | 97.15          | 98.76                 | 94.45

##### Weather Challenge Dataset

Dataset    | BLEU  | TreeAcc(whole) | TreeAcc(no-discourse) | TreeAcc(discourse)
-------    | ----  | -------------- | --------------------- | ------------------
S2S-Tree   | 76.75 | 91.10          | 96.62                 | 83.3
S2S-Constr | 77.45 | 95.74          | 98.52                 | 91.61

##### E2E Dataset

Dataset    | BLEU  | TreeAcc(whole) | TreeAcc(no-discourse) | TreeAcc(discourse)
-------    | ----  | -------------- | --------------------- | ------------------
S2S-Tree   | 74.58 | 97.06          | 99.68                 | 95.28
S2S-Constr | 74.69 | 99.25          | 99.89                 | 97.78

### License

TreeNLG is released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode), see [LICENSE](LICENSE.md) for details.
