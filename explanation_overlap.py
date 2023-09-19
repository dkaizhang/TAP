import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from argparse import ArgumentParser
from src.data import CSVDataset, MNISTDataset, get_transform, load_labels
from src.explainer import Explainer
from src.wrapper import ModelWrapper

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='decoyMNIST or FMNIST_decoyed')
    parser.add_argument('--model_name', type=str, default=None, help='MLP, MNISTModel ...')
    parser.add_argument('--load_from', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--mask_threshold', type=float, default=None, help='Set threshold to binarise explanations')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=64, help='Dataloading batch size')
    parser.add_argument('--method', type=str, default='input_gradients', help='input_gradients...')
    parser.add_argument('--print', action='store_true', default=False, help='whether to print charts or not')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--log', type=str, default='results/overlap.csv', help='path to log file')
    parser.add_argument('--stop_after', type=int, default=-1, help='Stop after this number of batches')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')

    return parser.parse_args()

def main(args):

    if args.dataset == 'decoyMNIST':
        path = 'data/MNIST/decoyed_mnist.npz'
        transform = get_transform('MNIST')
        colour = False
    elif args.dataset == 'pneu_text' or args.dataset == 'pneu_text_RGB':
        path = 'data/pneu/chest_xray/train_text/train_data.csv'
        decoy_path = 'data/pneu/chest_xray/train_text/decoy_positions.npy'
        transform = get_transform('pneu')
        colour = False
    elif args.dataset == 'knee_text' or args.dataset == 'knee_text_RGB':
        path = 'data/kneeKL224/train_decoyed.csv'
        decoy_path = 'data/kneeKL224/decoy_positions.npy'
        transform = get_transform('knee')
        colour = False


    else:
        print("unsupported")
        exit(1)

    print(args.dataset)

    if (args.dataset != 'pneu_text' and args.dataset != 'pneu_text_RGB') and (args.dataset != 'knee_text' and args.dataset != 'knee_text_RGB'):
        files = np.load(path)
        train_images = files['train_images']
        train_labels = files['train_labels']
        decoy_positions_train = files['decoy_positions_train']
        data = MNISTDataset(train_images, train_labels, transform, colour=colour)
    else:
        if args.dataset == 'pneu_text_RGB' or args.dataset == 'knee_text_RGB':
            data = CSVDataset(path, transform, convert=True)
        else:
            data = CSVDataset(path, transform, convert=False)
        decoy_positions_train = np.load(decoy_path)

    print("Decoy positions: ",decoy_positions_train.shape)

    if decoy_positions_train.ndim == 4 and decoy_positions_train.shape[3] == 3:
        print('max out colour channel in decoy positions')
        decoy_positions_train = decoy_positions_train.max(axis=-1)
        print("Decoy positions: ",decoy_positions_train.shape)

    labels = load_labels(args.dataset)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = ModelWrapper(model_name=args.model_name, device=device, load_from=args.load_from, labels=labels)

    explainer = Explainer(batch_size=args.batch_size, num_workers=args.num_workers, method=args.method)
    exp = explainer.explain(model.model, data, stop_after=args.stop_after).detach().numpy()

    # in case we stop early adjust length
    decoy_positions_train = decoy_positions_train[:len(exp)]

    assert(len(decoy_positions_train) == len(exp))

    exp = np.abs(exp).sum(axis=-3) # (N,28,28) getting rid of colour channel
    print("Explanations shape after summing: ", exp.shape)

    mask_threshold = args.mask_threshold
    mask_threshold = torch.quantile(a=torch.Tensor(exp).flatten(start_dim=1), q=mask_threshold, dim=1).numpy()
    print("Mask thresholds: ", mask_threshold.shape)

    # mask IGs greater than threshold, so that only the weak activations remain
    mask = torch.where(torch.abs(torch.Tensor(exp)) > torch.Tensor(mask_threshold)[:,None,None], 0, 1).numpy() # should be (N,28,28)
    print("Mask shape: ", mask.shape)

    # weak activations should coincide with decoy positions
    overlap = decoy_positions_train * mask # should be (N,28,28)

    coverage = overlap.sum(axis=(1,2)) / decoy_positions_train.sum(axis=(1,2))
    print("Average coverage (decoy area as base): ", coverage.mean())

    precision = overlap.sum(axis=(1,2)) / mask.sum(axis=(1,2))
    print("Average precision (mask area as base): ", precision.mean())

    if args.print:
        fig, axes = plt.subplots(1, 2)
        counts, edges = np.histogram(coverage, bins=[i/10 for i in range(11)])
        axes[0].stairs(counts, edges, fill=True)
        counts, edges = np.histogram(precision, bins=[i/10 for i in range(11)])
        axes[1].stairs(counts, edges, fill=True)

        plt.tight_layout()
        plt.show()

    return coverage.mean(), precision.mean()

if __name__ == '__main__':
    args = parse_args()

    if args.config is None:
        coverage, precision = main(args)

        output = pd.DataFrame([
                    {'dataset' : args.dataset,
                    'model_name' : args.model_name,
                    'load_from' : args.load_from,
                    'mask_threshold' : args.mask_threshold,
                    'method' : args.method,
                    'precision' : precision,
                    'coverage' : coverage     
                    }])

        if os.path.exists(args.log):
            log = pd.read_csv(args.log, index_col=0)
            log = pd.concat([log, output], ignore_index=True)
        else:
            log = output
        log.to_csv(args.log)

    else:
        configs = pd.read_csv(args.config).dropna(how='all').to_dict('records')
        for i, config in enumerate(configs):
            print(f"=== Experiment {i} / {len(configs)}===")
            args.dataset = config["dataset"]
            args.model_name = config["model_name"]
            args.load_from = os.path.join(config["runs_dir"],config["load_from"],"last-model.pt")
            args.mask_threshold = config["mask_threshold"]
            args.method = config["method"]
            coverage, precision = main(args)
            config["coverage"] = coverage
            config["precision"] = precision

            config = pd.DataFrame([config])
            if os.path.exists(args.log):
                log = pd.read_csv(args.log, index_col=0)
                log = pd.concat([log, config], ignore_index=True)
            else:
                log = config
            log.to_csv(args.log)
