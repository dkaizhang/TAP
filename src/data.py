import gzip
import numpy as np
import os
import pandas as pd
import random
import torch
import torchvision.datasets as datasets

from PIL import Image
from src.util import choose_split, sample_idx, train_val_split
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T
from urllib.request import urlretrieve

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform, colour=False, convert=False):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        self.colour = colour
        self.convert = convert

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], int(self.labels[idx])

        if self.colour:
            image = Image.fromarray(image, mode='RGB')
        else:
            image = Image.fromarray(image, mode='L')

        if self.convert:
            image = image.convert('RGB')

        image = self.transform(image)
            
        return image, label

class CSVDataset(Dataset):
    def __init__(self, data_csv, transform, convert=False):
        super().__init__()
        self.data_csv = pd.read_csv(data_csv)
        self.transform = transform
        self.convert = convert

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        image_file = self.data_csv.iloc[idx]['img_path']
        with Image.open(image_file) as img:

            if self.convert:
                img = img.convert('RGB')

            image = self.transform(img)
        label = int(self.data_csv.iloc[idx]['label'])
        
        return image, label
    
# explanation augmented dataset 
class XAugDataset(Dataset):
    def __init__(self, dataset, explanations):
        super().__init__()
        self.dataset = dataset
        self.explanations = explanations
        assert len(dataset) == len(explanations)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return *self.dataset.__getitem__(idx), self.explanations[idx]
   
def get_transform(data='MNIST'):

    if data == 'MNIST':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.1307,), (0.3081,))
            ]
        )  
    elif data == 'pneu':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.5711,),(0.1523,) # hand calc
                )
            ]
        )    
    elif data == 'knee':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.5549,),(0.2232,) # hand calc
                )
            ]
        )    
    else: 
        print("Data choice invalid, exiting...")
        exit(1)

    return transform

# Loads the MNIST dataset as Datasets
def load_MNIST(split, seed, frac, val_split):
    
    if frac is None:
        frac = 1.0
    else:
        frac = frac[0]

    path = 'data/MNIST/original_mnist.npz'
    data = np.load(path)

    train_data = MNISTDataset(data['train_images'], data['train_labels'], transform=get_transform('MNIST'))
    val_data = MNISTDataset(data['val_images'], data['val_labels'], transform=get_transform('MNIST'))
    test_data = MNISTDataset(data['test_images'], data['test_labels'], transform=get_transform('MNIST'))
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))
    
    return choose_split(split, train_data, val_data, test_data)

# Loads the Decoy MNIST dataset as Datasets
def load_decoyMNIST(split, seed, frac, val_split):

    if frac is None:
        frac = 1.0
    else:
        frac = frac[0]

    path = 'data/MNIST/decoyed_mnist.npz'
    data = np.load(path)

    train_data = MNISTDataset(data['train_images'], data['train_labels'], transform=get_transform('MNIST'))
    val_data = MNISTDataset(data['val_images'], data['val_labels'], transform=get_transform('MNIST'))
    test_data = MNISTDataset(data['test_images'], data['test_labels'], transform=get_transform('MNIST'))
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))

    # print('saving training instances')
    # temp_file = 'instances.npy'
    # np.save(temp_file, train_data.indices)

    return choose_split(split, train_data, val_data, test_data)

def load_pneu(decoy, split, seed, frac, convert=False):

    if frac is None:
        frac = 1.0
    else:
        frac = frac[0]

    transform = get_transform(data='pneu')

    if decoy == None:
        train_data = CSVDataset(data_csv='data/pneu/chest_xray/train/train_data.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/pneu/chest_xray/val/val_data.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test/test_data.csv',transform=transform, convert=convert)
    elif decoy == 'text':
        train_data = CSVDataset(data_csv='data/pneu/chest_xray/train_text/train_data.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/pneu/chest_xray/val_text/val_data.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_text/test_data.csv',transform=transform, convert=convert)
    elif decoy == 'stripe':
        train_data = CSVDataset(data_csv='data/pneu/chest_xray/train_stripe/train_data.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/pneu/chest_xray/val_stripe/val_data.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_stripe/test_data.csv',transform=transform, convert=convert)
    else:
        print('not implemented')
        exit(0)
    
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))

    return choose_split(split, train_data, val_data, test_data)

def load_knee(decoy, split, seed, frac, convert=False):

    if frac is None:
        frac = 1.0
    else:
        frac = frac[0]

    transform = get_transform(data='knee')

    if decoy == None:
        train_data = CSVDataset(data_csv='data/kneeKL224/train.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/kneeKL224/val.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/kneeKL224/test.csv',transform=transform, convert=convert)
    elif decoy == 'text':
        train_data = CSVDataset(data_csv='data/kneeKL224/train_decoyed.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/kneeKL224/val_decoyed.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/kneeKL224/test_decoyed.csv',transform=transform, convert=convert)
    elif decoy == 'stripe':
        train_data = CSVDataset(data_csv='data/kneeKL224/train_stripe.csv',transform=transform, convert=convert)
        val_data = CSVDataset(data_csv='data/kneeKL224/val_stripe.csv',transform=transform, convert=convert)
        test_data = CSVDataset(data_csv='data/kneeKL224/test_stripe.csv',transform=transform, convert=convert)
    else:
        print('not implemented')
        exit(0)
    
    if frac < 1.0:
        train_data = Subset(train_data, sample_idx(len(train_data), frac, seed))
        val_data = Subset(val_data, sample_idx(len(val_data), frac, seed))
        test_data = Subset(test_data, sample_idx(len(test_data), frac, seed))

    return choose_split(split, train_data, val_data, test_data)

def load_test_cor(data):

    if data == 'pneu_text_cor_RGB':
        transform = get_transform(data='pneu')
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_text_cor/test_cor_data.csv',transform=transform, convert=True)
    elif data == 'pneu_stripe_cor_RGB':
        transform = get_transform(data='pneu')
        test_data = CSVDataset(data_csv='data/pneu/chest_xray/test_stripe_cor/test_cor_data.csv',transform=transform, convert=True)

    elif data == 'knee_text_cor_RGB':
        transform = get_transform(data='knee')
        test_data = CSVDataset(data_csv='data/kneeKL224/test_cor_decoyed.csv',transform=transform, convert=True)
    elif data == 'knee_stripe_cor_RGB':
        transform = get_transform(data='knee')
        test_data = CSVDataset(data_csv='data/kneeKL224/test_cor_stripe.csv',transform=transform, convert=True)

    elif data == 'decoyMNIST_cor':
        data = np.load('data/MNIST/decoyed_mnist.npz')
        transform = get_transform(data='MNIST')
        test_data = MNISTDataset(data['test_cor_images'], data['test_cor_labels'], transform=transform)

    return test_data

def load_data(data, split, seed, frac, val_split):

    if seed is None:
        seed = random.randint(0,1000)

    if data == 'decoyMNIST':
        return load_decoyMNIST(split=split, seed=seed, frac=frac, val_split=val_split) 
    elif data == 'decoyMNIST_cor':
        return load_test_cor(data=data)

    elif data == 'MNIST':
        return load_MNIST(split=split, seed=seed, frac=frac, val_split=val_split)
    
    elif data == 'pneu':
        return load_pneu(decoy=None, split=split, seed=seed, frac=frac)
    elif data == 'pneu_RGB':
        return load_pneu(decoy=None, split=split, seed=seed, frac=frac, convert=True)

    elif data == 'pneu_text':
        return load_pneu(decoy='text', split=split, seed=seed, frac=frac)
    elif data == 'pneu_text_RGB':
        return load_pneu(decoy='text', split=split, seed=seed, frac=frac, convert=True)
    elif data == 'pneu_text_cor_RGB':
        return load_test_cor(data=data)

    elif data == 'pneu_stripe':
        return load_pneu(decoy='stripe', split=split, seed=seed, frac=frac)
    elif data == 'pneu_stripe_RGB':
        return load_pneu(decoy='stripe', split=split, seed=seed, frac=frac, convert=True)
    elif data == 'pneu_stripe_cor_RGB':
        return load_test_cor(data=data)
    
    elif data == 'knee_RGB':
        return load_knee(decoy=None, split=split, seed=seed, frac=frac, convert=True)
    
    elif data == 'knee_text_RGB':
        return load_knee(decoy='text', split=split, seed=seed, frac=frac, convert=True)
    elif data == 'knee_text_cor_RGB':
        return load_test_cor(data=data)
    
    elif data == 'knee_stripe_RGB':
        return load_knee(decoy='stripe', split=split, seed=seed, frac=frac, convert=True)
    elif data == 'knee_stripe_cor_RGB':
        return load_test_cor(data=data)

    else:
        print("Unsupported data, exiting...")
        exit(1) 

def load_labels(data):
    mnist_labels = [i for i in range(10)]
    pneumnist_labels = [0,1]
    knee_labels = [i for i in range(5)]

    if data[0:10] == 'decoyMNIST' or data == 'MNIST':
        return mnist_labels
    elif data[0:4] == 'pneu':
        return pneumnist_labels
    elif data[0:4] == 'knee':
        return knee_labels
    else:
        print("Unsupported data, exiting...")
        exit(1) 

# Loads explanations and returns numpy arrays
def load_explanations(path, data):

    pneu_size = 224
    knee_size = 224

    cached = np.load(path)
    if data == 'MNIST' or data[0:10] == 'decoyMNIST':
        expl = np.reshape(cached, (-1,1,28,28))
    elif data == 'pneu_RGB' or data == 'pneu_text_RGB' or data == 'pneu_stripe_RGB' or data == 'pneu_text_cor_RGB' or data == 'pneu_stripe_cor_RGB':
        expl = np.reshape(cached, (-1,3,pneu_size,pneu_size))
    elif data[0:4] == 'pneu':
        expl = np.reshape(cached, (-1,1,pneu_size,pneu_size))
    elif data == 'knee_RGB' or data == 'knee_text_RGB' or data == 'knee_stripe_RGB' or data == 'knee_text_cor_RGB' or data == 'knee_stripe_cor_RGB':
        expl = np.reshape(cached, (-1,3,knee_size,knee_size))
    else:
        print("invalid data - don't know how to reshape the explanations")
        exit(1)
    return expl

def load_weights(data):

    if data[0:4] == 'pneu':
        num_samples = 4708
        num_class = [1214, 3494]
        weights = torch.Tensor( [num_samples / num_c for num_c in num_class])
        return weights / sum(weights)        
    else:
        return None