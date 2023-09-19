import numpy as np
import os
import shutil
import torch
import yaml

from src.args import parse_config
from src.explainer import Explainer
from src.wrapper import ModelWrapper
from src.data import load_data, load_labels

def main(args):

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
    print('device:', device)

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    # TO DO make sure this happens with the other scripts
    experiment_folder = os.path.join('experiments', config['experiment'])
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    shutil.copy(args.config, experiment_folder)

    data = load_data(data=config['data'], split=config['split'], seed=config['seed'], frac=config['frac'], val_split=config['val_split'])
    labels = load_labels(data=config['data'])

    for i, teacher in enumerate(config['teacher_load_froms']):

        model = ModelWrapper(model_name=config['teacher_model_names'][i], device=device, task_loss=config['task_loss'], load_from=teacher, labels=labels)

        explainer = Explainer(batch_size=config['batch_size'], num_workers=config['num_workers'], method=config['method'], device=device)

        teacher_folder = os.path.join(experiment_folder, f'teacher_{i}')
        if not os.path.exists(teacher_folder):
            os.mkdir(teacher_folder)

        save_to = os.path.join(teacher_folder, f"explain_{config['split']}.npy") 
        explanations = explainer.explain(model.model, data).detach().numpy()
        np.save(save_to, explanations)


if __name__ == "__main__":
    args = parse_config()
    main(args)