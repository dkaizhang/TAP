import pandas as pd
import torch

from src.args import parse_train_args
from src.data import load_data, load_labels, load_weights
from src.trainer import Trainer
from src.util import listify_keys, pathify_list, get_summarywriter
from src.wrapper import ModelWrapper

def main(args):

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print('device:', device)
    
    labels = load_labels(data=args.data)
    weights = load_weights(data=args.data)

    model = ModelWrapper(model_name=args.model_name, load_from=args.load_from, pretrained=args.pretrained, lr=args.lr, l2=args.l2, device=device, task_loss=args.task_loss, weights=weights, labels=labels)

    train_data = load_data(args.data, split=args.split, seed=args.seed, frac=args.frac, val_split=args.val_split)
    val_data = load_data(args.data, split=args.val_on, seed=args.seed, frac=args.frac, val_split=args.val_split)
    print('Validation set:', args.val_on)

    # if hasattr(train_data, 'indices') and hasattr(val_data, 'indices'):
    #     assert set(train_data.indices).isdisjoint(set(val_data.indices))  

    print(f"Train {len(train_data)}, val {len(val_data)}")

    trainer = Trainer(batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        num_workers=args.num_workers,
                        writer = get_summarywriter(args.out_dir))
    loss_dict = trainer.fit(model, train_data, val_data, save_every=args.save_every)
    return loss_dict

if __name__ == "__main__":
    args = parse_train_args()
    if args.batch_configs is not None:
        batch_configs = pd.read_csv(args.batch_configs).dropna(how='all').to_dict('records')
        for i, config in enumerate(batch_configs):
            print(f"=== Pretraining {1+i} / {len(batch_configs)} ===")
            args.model_name = config['model_name']
            args.load_from = listify_keys(config, "load_from", 1, str) # hacky way of dealing with nans in the csv
            if isinstance(args.load_from, list): # necessary in case the above returned None
                args.load_from = pathify_list(args.load_from, config['runs_dir'], "last-model.pt")[0]            
            args.pretrained = config['pretrained']
            args.out_dir = config['out_dir']
            args.batch_size = int(config['batch_size'])
            args.lr = config['lr']
            args.data = config['data']
            args.split = config['split']
            args.val_on = config['val_on']
            args.seed = int(config['seed'])
            args.frac = listify_keys(config, "frac", 5, float) 
            loss_dict = main(args)
            del loss_dict
            torch.cuda.empty_cache()

    else:
        loss_dict = main(args)