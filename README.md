# SNN4SA: Spiking Neural Network for Sentiment Analysis.

**Note**: 

- Detailed introduction and analysis are provided in `ExperimentalReport.pdf`.
- If you have any problems, feel free to contact us.

### Requirements

**Please use Python 3.7+** To get started:

```shell
git clone https://github.com/WanliYoung/SNN4SA.git
conda create -n SNN4SA python=3.7
...
pip install -r requirements.txt
```

### Convert Pre-trained Word Embeddings

To make the pre-trained word embeddings available for SNN, we need to convert them into values in [0, 1].

Prepare the `glove.6B.300d.txt` and `SST2` dataset in `data`. Then you can utilize the `data_process.py` to gain converted word embeddings.

### CNN

```shell
python train.py --mode train --model_mode ann --model_type textcnn
```

### SNN

```shell
python train.py --mode train --model_mode snn --model_type lstm
```

### Conversion + Fine-tuning SNN

```shell
python train.py --mode conversion --model_mode snn --model_type textcnn --conversion_mode tune
```

### CNN + SNN

```shell
python train.py --mode train --model_mode combine --model_type cnn_snn
```

