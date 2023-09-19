from argparse import ArgumentParser

def parse_train_args():

    parser = ArgumentParser()
    parser.add_argument('--batch_configs', type=str, default=None, help="Path to a file with pretrain configs")
    parser.add_argument('--epochs', type=int, default=0, help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=0, help='Save model every x epochs')
    parser.add_argument('--data', type=str, default='decoyMNIST', help='Choose dataset from MNIST, decoyMNIST...')
    parser.add_argument('--split', type=str, default='train', help='Choose train, val or test to train on')
    parser.add_argument('--frac', nargs='+', type=float, default=None, help="Fraction of dataset to be used")    
    parser.add_argument('--val_on', type=str, default='val', help="Choose train, val or test to val on")    
    parser.add_argument('--val_split', type=float, default=0.1, help="Fraction of train to use for validation") 
    parser.add_argument('--task_loss', type=str, default='cross_entropy', help="Choose cross_entropy")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")    
    parser.add_argument('--l2', type=float, default=0.0001, help="L2 regulariser")    
    parser.add_argument('--model_name', type=str, help="Architecture choices")
    parser.add_argument('--load_from', type=str, default=None, help='Model checkpoint to continue training from')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained if available')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=32, help='Dataloading batch size')
    parser.add_argument('--seed', type=int, default=None, help='Manually set a seed')
    parser.add_argument('--gpu', type=int, default=0, help='Choose gpu if available')
    parser.add_argument('--out_dir', type=str, default='runs', help="Output folder for all trained models")
    return parser.parse_args()

def parse_evaluate_args():

    parser = ArgumentParser()
    parser.add_argument('--batch_configs', type=str, default=None, help="Path to a file with evaluation configs")
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model checkpoint')
    parser.add_argument('--model_name', type=str, default=None, help="Architecture choices")
    parser.add_argument('--print_confusion', action='store_true', default=False, help='Set to print out confusion matrix')
    parser.add_argument('--log', type=str, default=None, help='Path to a log file')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=32, help='Dataloading batch size')
    parser.add_argument('--data', type=str, default='decoyMNIST', help='Choose dataset from MNIST, decoyMNIST...')
    parser.add_argument('--frac', nargs='+', type=float, default=None, help="Fraction of dataset to be used")    
    parser.add_argument('--split', type=str, default='test', help='Choose train, val or test to evaluate on')
    parser.add_argument('--val_split', type=float, default=0.1, help="Fraction of train to use for validation")    
    parser.add_argument('--task_loss', type=str, default='cross_entropy', help="Choose cross_entropy")
    parser.add_argument('--seed', type=int, default=None, help='Manually set a seed')
    parser.add_argument('--gpu', type=int, default=0, help='Choose gpu if available')
    return parser.parse_args()

def parse_config():

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    return parser.parse_args()

def parse_batch_config():

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--batch_configs', type=str, required=True, help='Path to configs csv')
    parser.add_argument('--config_folder', type=str, default='batch_configs', help='A folder to save configs in')
    parser.add_argument('--loss_log', type=str, default=None, help="A csv file to which losses are written")

    return parser.parse_args()