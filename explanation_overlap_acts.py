import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from src.data import CSVDataset, MNISTDataset, get_transform, load_labels, XAugDataset
from src.explainer import Explainer
from src.wrapper import ModelWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='decoyMNIST or FMNIST_decoyed')
    parser.add_argument('--model_name', type=str, default=None, help='MLP, MNISTModel ...')
    parser.add_argument('--load_from', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--mask_threshold', type=float, default=None, help='Set threshold to binarise explanations')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloading batch size')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--log', type=str, default='results/top_acts.csv', help='path to log file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')

    return parser.parse_args()

def main(args):

    if args.dataset == 'decoyMNIST':
        path = 'data/MNIST/decoyed_mnist.npz'
        transform = get_transform('MNIST')
        colour = False
    elif args.dataset == 'pneu_text' or args.dataset == 'pneu_text_RGB':
        path = 'data/pneu/chest_xray/test_text/test_data.csv'
        decoy_path = 'data/pneu/chest_xray/test_text/decoy_positions_test.npy'
        transform = get_transform('pneu')
        colour = False
    elif args.dataset == 'knee_text' or args.dataset == 'knee_text_RGB':
        path = 'data/kneeKL224/test_decoyed.csv'
        decoy_path = 'data/kneeKL224/decoy_positions_test.npy'
        transform = get_transform('knee')
        colour = False
    else:
        exit(1)

    print(args.dataset)

    # need to make this test
    if (args.dataset != 'pneu_text' and args.dataset != 'pneu_text_RGB') and (args.dataset != 'knee_text' and args.dataset != 'knee_text_RGB'):
        files = np.load(path)
        test_images = files['test_images']
        test_labels = files['test_labels']
        decoy_positions = files['decoy_positions_test']
        data = MNISTDataset(test_images, test_labels, transform, colour=colour)
    else:
        if args.dataset == 'pneu_text_RGB' or args.dataset == 'knee_text_RGB':
            data = CSVDataset(path, transform, convert=True)
        else:
            data = CSVDataset(path, transform, convert=False)
        decoy_positions = np.load(decoy_path)
    if decoy_positions.ndim == 4 and decoy_positions.shape[3] == 3:
        print('max out colour channel in decoy positions')
        decoy_positions = decoy_positions.max(axis=-1)
        print("Decoy positions: ",decoy_positions.shape)
    print("Decoy positions: ",decoy_positions.shape)

    data = XAugDataset(data, decoy_positions)
    labels = load_labels(args.dataset)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    model = ModelWrapper(model_name=args.model_name, device=device, load_from=args.load_from, labels=labels)
    model.model.to(device)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    coverages = {0:[], 1:[], 2:[], 3:[]}
    precisions = {0:[], 1:[], 2:[], 3:[]}
    for batch_idx, data in enumerate(tqdm(loader)):
        x, y, decoy_positions = data
        x = x.to(device)
        y = y.to(device)
        decoy_positions = decoy_positions.to(device)
        y_hat, acts = model.model(x)

        for i, act in enumerate(acts):
            # print('need to match decoy pos size: ', decoy_positions.shape)
            act = act.sum(dim=1)
            # print('act shape after summing out channels: ', act.shape)
            resized_act = F.interpolate(act[:,None,:,:], size=decoy_positions.shape[1]).squeeze()
            # print("resized act to: ", resized_act.shape)
            mask_threshold = args.mask_threshold
            mask_threshold = torch.quantile(a=torch.abs(resized_act).flatten(start_dim=1), q=mask_threshold, dim=1)
            # print("Mask thresholds: ", mask_threshold.shape)
            # print(mask_threshold)
            # high acts will be highlighted
            mask = torch.where(torch.abs(resized_act) > mask_threshold[:,None,None], 1, 0) # should be (N,28,28)
            # print("Mask shape: ", mask.shape)
            overlap = decoy_positions * mask
            # print("overlap: ", overlap.shape)
            coverage = overlap.sum(axis=(1,2)) / decoy_positions.sum(axis=(1,2))
            coverages[i].append(coverage.detach().cpu())
        
            precision = overlap.sum(axis=(1,2)) / mask.sum(axis=(1,2))
            precisions[i].append(precision.detach().cpu())
        del x
        del acts

    for i in range(4):
        if len(coverages[i]) > 0: 
            coverages[i] = torch.cat(coverages[i]).mean().item()
            precisions[i] = torch.cat(precisions[i]).mean().item()

    return coverages, precisions

if __name__ == '__main__':
    args = parse_args()

    if args.config is None:
        coverages, precisions = main(args)

        output = pd.DataFrame([
                    {'dataset' : args.dataset,
                    'model_name' : args.model_name,
                    'load_from' : args.load_from,
                    'mask_threshold' : args.mask_threshold,
                    'precision_0' : precisions[0],
                    'coverage_0' : coverages[0],     
                    'precision_1' : precisions[1],
                    'coverage_1' : coverages[1],     
                    'precision_2' : precisions[2],
                    'coverage_2' : coverages[2],     
                    'precision_3' : precisions[3],
                    'coverage_3' : coverages[3]     
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
            print(f"=== Experiment {1+i} / {len(configs)}===")
            args.dataset = config["dataset"]
            args.model_name = config["model_name"]
            args.load_from = os.path.join(config["runs_dir"],config["load_from"],"last-model.pt")
            args.mask_threshold = config["mask_threshold"]
            coverages, precisions = main(args)
            config["coverage_0"] = coverages[0]
            config["precision_0"] = precisions[0]
            config["coverage_1"] = coverages[1]
            config["precision_1"] = precisions[1]
            config["coverage_2"] = coverages[2]
            config["precision_2"] = precisions[2]
            config["coverage_3"] = coverages[3]
            config["precision_3"] = precisions[3]

            config = pd.DataFrame([config])
            if os.path.exists(args.log):
                log = pd.read_csv(args.log, index_col=0)
                log = pd.concat([log, config], ignore_index=True)
            else:
                log = config
            log.to_csv(args.log)

            with torch.cuda.device(f"cuda:0"):
                print(f"emptying cuda")
                torch.cuda.empty_cache()