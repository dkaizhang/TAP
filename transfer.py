# this has to be run after a debate
import torch
import yaml

from src.args import parse_config
from src.data import load_data, load_labels, load_weights, XAugDataset
from src.picker import Picker
from src.trainer import Trainer
from src.util import get_summarywriter
from src.wrapper import TransferModelWrapper

def main(args):

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        
    picker = Picker(config)
    explanations = picker.get_winners()
    print(f"Explanations {len(explanations)}")

    device = f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
    print('device:', device)

    labels = load_labels(data=config['data'])
    weights = load_weights(data=config['data'])
    model = TransferModelWrapper(model_name=config['model_name'], 
                        load_from=config['load_from'],
                        pretrained=config['pretrained'],
                        teacher_model_name=config['teacher_model_names'][0],
                        teacher_load_from=config['teacher_load_froms'][0],
                        kd_lambda=config['kd_lambda'],
                        temperature=config['temperature'],
                        et_lambda=config['et_lambda'],
                        at_lambda=config['at_lambda'],
                        l2=config['l2'],
                        lr=config['lr'],
                        weights=weights,
                        device=device, 
                        task_loss=config['task_loss'],
                        labels=labels)

    train_data = load_data(config['data'], split=config['split'], seed=config['seed'], frac=config['frac'], val_split=config['val_split'])
    train_data = XAugDataset(train_data, explanations)
    val_data = load_data(config['data'], split=config['val_on'], seed=config['seed'], frac=config['frac'], val_split=config['val_split'])
    print('Validation set:', config['val_on'])

    # if hasattr(train_data, 'indices'):
    #     assert set(train_data.indices).isdisjoint(set(val_data.indices))  
        
    print(f"Train {len(train_data)}, val {len(val_data)}")

    trainer = Trainer(batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        num_workers=config['num_workers'],
                        writer = get_summarywriter(config['out_dir']))
    loss_dict = trainer.fit(model=model, train_data=train_data, val_data=val_data, save_every=config['save_every'])
    return loss_dict

if __name__ == "__main__":
    args = parse_config()
    _ = main(args)