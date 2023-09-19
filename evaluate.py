import os
import pandas as pd
import torch

from pathlib import Path
from src.args import parse_evaluate_args
from src.data import load_data, load_labels
from src.trainer import Trainer
from src.util import listify_keys
from src.wrapper import ModelWrapper

def main(args):

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print('device:', device)

    data = load_data(data=args.data, split=args.split, seed=args.seed, frac=args.frac, val_split=args.val_split)
    labels = load_labels(data=args.data)

    if os.path.exists(args.model_path):
        model = ModelWrapper(model_name=args.model_name, load_from=args.model_path, device=device, task_loss=args.task_loss, labels=labels)
        print(f"Evaluate on {args.data} {args.split} dataset {len(data)}")

        trainer = Trainer(batch_size=args.batch_size, num_workers=args.num_workers)
        loss, confusion, perf = trainer.test(model, data)

        if args.print_confusion:
            print(confusion)

        output = pd.DataFrame([
                    {'folder' : Path(args.model_path).parent.name,
                    'data' : args.data,
                    'frac' : args.frac,
                    'acc' : round(confusion.diagonal().sum()/confusion.sum(),4),
                    'f1' : round(perf["f1"],4),
                    'auc' : round(perf["auc"],4),
                    'mse' : round(perf["mse"],4),
                    'loss' : round(loss.item(),4)     
                    }])

    else:
        print(f'{args.model_path} not found, skipping')
        output = pd.DataFrame([
            {'folder' : Path(args.model_path).parent.name,
            'data' : args.data,
            'frac' : args.frac,
            'acc' : -999,
            'f1' : -999,
            'auc' : -999,
            'mse' : -999,
            'loss' : -999     
            }])

    if args.log is not None:
        if os.path.exists(args.log):
            log = pd.read_csv(args.log, index_col=0)
            log = pd.concat([log, output], ignore_index=True)
        else:
            log = output
        log.to_csv(args.log)

if __name__ == "__main__":
    args = parse_evaluate_args()
    if args.batch_configs is not None:
        batch_configs = pd.read_csv(args.batch_configs).dropna(how='all').to_dict('records')
        for i, config in enumerate(batch_configs):
            print(f"=== Evaluation {1+i} / {len(batch_configs)} ===")
            args.model_name = config['model_name']
            args.model_path = str(os.path.join(config['runs_dir'], config['model_dir'], config['checkpoint']))
            args.batch_size = int(config['batch_size'])
            args.data = config['data']
            args.split = config['split']
            args.frac = listify_keys(config, "frac", 5, float) 
            args.seed = int(config['seed'])
            main(args)
    else:
        main(args)