# SAM-AntiSpoofing
This is the official implementation of [our paper](https://arxiv.org/pdf/2506.11532)

```bibtex
From Sharpness to Better Generalization for Speech Deepfake Detection
by Wen Huang, Xuechen Liu, Xin Wang, Junichi Yamagishi, and Yanmin Qian
Accepted by Interspeech 2025
```

This work investigates sharpness as a theoretical proxy for generalization in SDD and applies Sharpness-Aware Minimization (SAM) for better and more stable performance across diverse unseen test sets.



## Data Preparation
Our experiments utilize the following datasets:
- 19LA: [ASVspoof2019 LA](https://www.asvspoof.org/index2019.html)
- 21LA & 21DF: [ASVspoof2021 LA & DF](https://www.asvspoof.org/index2021.html)
- ITW: [In_The_Wild](https://deepfake-total.com/in_the_wild)
- FOR: [FakeOrReal](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- WF: [WaveFake](https://zenodo.org/records/5642694)
- ADD: [ADD2022 evaluation set](https://zenodo.org/records/12188055)
- SC: [SpoofCeleb evaluation set](https://www.jungjee.com/spoofceleb/)

For details on dataset structure, required audio folder organization, and metadata formatting, please refer to data/README.md.


## Environment Setup
Clone the specific version of Fairseq:
[Fairseq Repository](https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)

Install Fairseq in editable mode:
```bash
pip install --editable ./
```

Install required dependencies:
```bash
pip install -r requirements.txt
```


## Training and Evaluation
Train (single or multiple gpu on slurm):
```bash
python train.py --config configs/config.yaml --exp_dir exp/debug
```

Evaluate (single gpu):
```bash
python evaluate.py --exp_dir exp/debug --epoch best --eval 19LA
# Or batch evaluation
bash scripts/evaluate.sh
```

## Analysis and Visualization

Compute sharpness (single gpu):
```bash
python compute_sharp.py --exp_dir exp/debug --epoch best --eval 19LA
# Or batch evaluation
bash scripts/sharpness.sh
```

Compute sharpness and eer for subsets (single gpu):
```bash
python evalute_sharp.py --exp_dir exp/debug --epoch best --test 1
```

Plot loss landscape (single gpu):
```bash
python plot_loss_landscape.py --exp_dir exp/debug --epoch best
```

## Reference Repo
Thanks for the following repos:
1. [aasist](https://github.com/clovaai/aasist): AASIST
2. [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing): Wav2vec2+AASIST & Rawboost
3. [sam](https://github.com/davda54/sam): SAM optimizer
4. [loss-landscape](https://github.com/marcellodebernardi/loss-landscapes): Loss landscape visualization

## Cite
If you find this code useful, please consider citing our paper:
```bibtex
@inproceedings{sam_spoofing,
  author={Huang, Wen and Liu, Xuechen and Wang, Xin and Yamagishi, Junichi and Qian, Yanmin},
  title={From Sharpness to Better Generalization for Speech Deepfake Detection},
  year=2025,
  booktitle={Proc. Interspeech (to appear)},
}
```

## Acknowledgements
This work was conducted during the first author’s internship at NII, Japan. 
This study is partially supported by JST AIP Acceleration Research (JPMJCR24U3). 
This work was also supported in part by China NSFC projects under Grants 62122050 and 62071288, 
in part by Shanghai Municipal Science and Technology Commission Project under Grant 2021SHZDZX0102.

