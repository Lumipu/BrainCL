# BrainCL

Code for the paper:

**BrainCL: Transformer-Based Brain Network Contrastive Learning with Multi-Order Topology and Salience Masking (IEEE TMI)**

## 1. Project Structure

The main project files are organized as follows:

```text
BrainCL/
├── conf/
│   ├── config.ini                 # Dataset directory configuration
│   └── loadpath.py                # Dataset path construction
├── data/
│   └── ABIDE_TOP6/
│       └── sFCN.h5                # FCN matrices, labels, and sites
├── dataloader/
│   └── mydataloader.py             # HDF5 loading
├── model/
│   └── brainclmodel.py             # Training, validation, and testing
├── network/
│   └── MoTFormer.py                # MoTFormer architecture
├── utils/
│   ├── losses.py                  # Contrastive loss functions
│   ├── masker.py                  # Salience-informed ROI masking
│   └── meter.py                   # Metric aggregation
├── script/
│   ├── train_online.sh
│   └── train_offline.sh
├── log/
│   ├── logger.conf                # Logging configuration
│   ├── mylogger.py                # Logging interface
│   ├── record/
│   └── terminal/
├── train.py
└── README.md
```

## 2. Environment

- Run the launch scripts in a Linux environment with Bash.
- Before running a script, activate a Python environment with `torch`, `numpy`, `h5py`, and `scikit-learn` installed. The scripts use `python` from the active environment.
- The current model uses CUDA and requires an available NVIDIA GPU and a CUDA-enabled PyTorch installation.

## 3. Prepare the HDF5 Dataset

Store all samples in a single HDF5 file named `sFCN.h5`, placed in `data/ABIDE_TOP6/` under the project root:

```text
data/ABIDE_TOP6/sFCN.h5
```

The default configuration in `conf/config.ini` is:

```ini
[Data]
data_dir = ./data/ABIDE_TOP6
```

`conf/loadpath.py` appends `/sFCN.h5` to this directory to construct the dataset path. Placing the file at the location above requires no path changes.

### HDF5 Structure

Create the following three datasets at the root level of the HDF5 file. Their names are case-sensitive:

```text
sFCN.h5
├── fcn      (N, 200, 200)
├── label    (N,)
└── site     (N,)
```

Here, `N` is the number of samples.

| Field | Shape | Data type | Description |
| --- | --- | --- | --- |
| `fcn` | `(N, 200, 200)` | Floating-point values, preferably `float32` | Functional connectivity matrix for each sample; the current model uses the CC200 atlas |
| `label` | `(N,)` | Numeric values, either `0` or `1` | Binary label for each sample; converted to a floating-point tensor during training |
| `site` | `(N,)` | UTF-8-encoded byte strings | Acquisition site for each sample |

Data requirements:

- `fcn[i]`, `label[i]`, and `site[i]` must refer to the same sample. All three datasets must contain the same number of samples.
- Store `label` and `site` as one-dimensional arrays, not arrays with shape `(N, 1)`.
- Use the same ordering of brain regions for every sample.
- The site loader calls `site.decode('utf-8')`, so site values must be returned as decodable byte strings.
- Define the class mapping for `0` and `1` during data preparation and use it consistently.

### Saving Example

From the project root, call the following function with your prepared `fcns`, `labels`, and `sites` arrays.

```python
from pathlib import Path

import h5py
import numpy as np


def save_dataset(fcns, labels, sites):
    fcns = np.asarray(fcns, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    sites = np.asarray([site.encode('utf-8') for site in sites], dtype='S')

    assert fcns.ndim == 3 and fcns.shape[1:] == (200, 200)
    assert labels.shape == (len(fcns),)
    assert sites.shape == (len(fcns),)
    assert np.isfinite(fcns).all()
    assert np.isin(labels, [0, 1]).all()

    output = Path('data/ABIDE_TOP6/sFCN.h5')
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, 'w') as file:
        file.create_dataset('fcn', data=fcns)
        file.create_dataset('label', data=labels)
        file.create_dataset('site', data=sites)
```

## 4. Run Training

From the project root, create the log directories and change to the `script` directory:

```bash
mkdir -p log/record log/terminal
cd script
```

### Foreground Training: train_online.sh

```bash
bash train_online.sh
```

Training runs in the foreground and occupies the current terminal until it finishes. Press `Ctrl+C` to stop training.

Metrics are also saved to the following path under the project root:

```text
log/record/myrecord.log
```

### Background Training: train_offline.sh

```bash
bash train_offline.sh
```

This script launches training in the background using `nohup python -u train.py ... &`, allowing you to continue entering terminal commands.

Standard output and standard error are both redirected to:

```text
log/terminal/myrecord.log
```

The application's metric log is also written to:

```text
log/record/myrecord.log
```

The script names `online` and `offline` refer to foreground and background execution. Both use the same training pipeline and data.

## Citation

If you find this repo helpful, please cite our paper.

```bibtex
@article{Zhang2026BrainCL,
  title   = {BrainCL: Transformer-Based Brain Network Contrastive Learning With Multi-Order Topology and Salience Masking},
  author  = {Zhang, Yongliang and Qian, Haochen and Yang, Jinbo and Chen, Fangfang and Dai, Xi-Jian and Fan, Kaiyu and Xiao, Li and Wang, Yu-Ping},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {45},
  number  = {9},
  pages   = {4784--4796},
  year    = {2026}
}
```
