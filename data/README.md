## Data Overview

This folder contains the datasets, organized into audio files and corresponding metadata.

```
data/
├── audio/      # Directory for all dataset audio files
└── metadata/   # Directory for CSV metadata files

```

## Dataset Summary

| Dataset               | # Utterances | Notes                                                                     |
|-----------------------|--------------|---------------------------------------------------------------------------|
| ASVspoof2019_LA_train | 25,380       |                                                                           |
| ASVspoof2019_LA_dev   | 24,843       |                                                                           |
| ASVspoof2019_LA_eval  | 71,237       |                                                                           |
| ASVspoof2021_LA_eval  | 148,176      |                                                                           |
| ASVspoof2021_DF_eval  | 611,829      |                                                                           |
| In_The_Wild           | 31,779       |                                                                           |
| FakeOrReal            | 15,432       | Normalized version; includes val + test splits                            |
| WaveFake              | 53,405       | 40% from JSUT and LJSpeech; Original sizes: LJSPEECH=121,083; JSUT=14,971 |
| ADD2022_eval          | 126,862      | Track 3.2, Round 2 test sets                                              |
| SpoofCeleb_evaluation | 91,131       | Evaluation subset                                                         |

## Preprocessing

Example preprocessing script to generate metadata CSVs:

```bash
python data_process/create_meta.py
```