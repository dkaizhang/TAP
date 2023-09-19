# This describes how to run the scripts to reproduce the results from Targeted Activation Penalties Help CNNs Ignore Spurious Signals

## Project structure

The main code to train and evaluate models is in the root directory.
Data should be placed in a `data` directory.
Configs should be placed in a `configs` directory.

## Data

### MNIST

Run `python create_decoy_mnist.py` to download MNIST create the decoyed version of MNIST.

### PNEU

Download PNEU from [https://data.mendeley.com/datasets/rscbjbr9sj/3](https://data.mendeley.com/datasets/rscbjbr9sj/3) and place `chest_xray` in `data/pneu` which you may need to create.
Run `python create_pneu.py` to create PNEU with text and stripe artifacts.

### KNEE

Download KNEE from [https://data.mendeley.com/datasets/56rmx5bjcr/1](https://data.mendeley.com/datasets/56rmx5bjcr/1) and place kneeKL224 in `data`.
Run `python create_knee.py` to create KNEE with text and stripe artifacts.

## Training

To pre-train teacher models, use `pretrain.py`. Example usage:
```
python pretrain.py \
    --epochs 1 \
    --data pneu_RGB \
    --model_name resnet18 \
    --pretrained \
    --batch_size 16 \
    --num_workers 4 \
    --seed 0
```

To output explanations, fill out a `config.yaml` file (example provided at `configs/config.yaml`) and use `explain.py`. Example usage:
```
python explain.py --config configs/config.yaml
```

To train a student with XS, fill out a `config.yaml` file and use `teach.py`. Example usage:

```
python teach.py --config configs/config.yaml
```

To train a student with KD, fill out a `config.yaml` file and use `transfer.py`. Example usage:

```
python transfer.py --config configs/config.yaml
```

To train the models from the paper, run the scripts with seeds `0, 1, 2, 3, 4`. 
Note: additional argument options are listed in `src/args.py`.

## Evaluation

To evaluate the performance of a model, use `evaluate.py`
```
python evaluate.py \
    --model_path <path to model checkpoint> \
    --model_name resnet18 \
    --data pneu_text_RGB
```

To evaluate the overlap of the lowest input gradients and the spurious signals, use `explanation_overlap.py`. Example usage:
```
python explanation_overlap.py \
    --model_name resnet18 \
    --load_from <path to model checkpoint> \
    --dataset pneu_text_RGB \
    --mask_threshold 0.01 
```

To evaluate the overlap of the top input gradients and the spurious signals, use `explanation_overlap_ig.py`. Example usage:
```
python explanation_overlap_ig.py \
    --model_name resnet18 \
    --load_from <path to model checkpoint> \
    --dataset pneu_text_RGB \
    --mask_threshold 0.25
```

To evaluate the overlap of the top activations and the spurious signals, use `explanation_overlap_acts.py`. Example usage:
```
python explanation_overlap_acts.py \
    --model_name resnet18_activations \
    --load_from <path to model checkpoint> \
    --dataset pneu_text_RGB \
    --mask_threshold 0.25
```
