import gzip
import numpy as np
import os
import random

from argparse import ArgumentParser
from copy import deepcopy
from urllib.request import urlretrieve

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST')
    parser.add_argument('--targets', nargs='+', type=int, default=None, help='Labels to decoy')
    
    return parser.parse_args()

def read_gz_files(dir, x_file, y_file, image_size, num_images):

    with gzip.open(os.path.join(dir, x_file),'rb') as f:
        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        images = np.copy(np.frombuffer(buf, dtype=np.uint8))
        images = images.reshape(num_images, image_size, image_size)

    with gzip.open(os.path.join(dir, y_file),'rb') as f:
        f.read(8)
        buf = f.read(num_images)
        labels = np.copy(np.frombuffer(buf, dtype=np.uint8))
    
    return images, labels

def augment(images, labels, number_of_labels, image_size, patch_size, randomise, targets):

    step = 255 // number_of_labels

    decoy_positions = []
    decoyed_images = []
    for i, image in enumerate(images):
        offset_x = random.randint(0,1) * (image_size - patch_size)
        offset_y = random.randint(0,1) * (image_size - patch_size)

        decoy_present = True
        if not randomise:
            if targets is None or labels[i] in set(targets):
                value = 255 - labels[i] * step
            else:
                decoy_present = False
        else:
            value = 255 - random.randint(0, number_of_labels - 1) * step

        decoy_position = np.zeros_like(image)
        if decoy_present:
            decoy_position[offset_x : offset_x + patch_size, offset_y : offset_y + patch_size] = 1
        decoy_positions.append(decoy_position)
        
        decoyed_image = deepcopy(image)
        if decoy_present:
            decoyed_image[offset_x : offset_x + patch_size, offset_y : offset_y + patch_size] = value
        decoyed_images.append(decoyed_image)

    decoy_positions = np.asarray(decoy_positions)
    decoyed_images = np.asarray(decoyed_images)

    return decoyed_images, decoy_positions

def main(args):
    
    number_of_labels = 10

    train_x_file = 'train-images-idx3-ubyte.gz'
    train_y_file = 'train-labels-idx1-ubyte.gz'
    test_x_file = 't10k-images-idx3-ubyte.gz'
    test_y_file = 't10k-labels-idx1-ubyte.gz'

    dl_train_x = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    dl_train_y = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    dl_test_x = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    dl_test_y = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    dir = f'data/{args.dataset}'
    if not os.path.exists(dir):
        print(f'downloading {args.dataset}')
        os.makedirs(dir)
        urlretrieve(dl_train_x, os.path.join(dir,train_x_file))
        urlretrieve(dl_train_y, os.path.join(dir,train_y_file))
        urlretrieve(dl_test_x, os.path.join(dir,test_x_file))
        urlretrieve(dl_test_y, os.path.join(dir,test_y_file))

    image_size = 28
    patch_size = 4
    random.seed(args.seed)

    train_images, train_labels = read_gz_files(dir, train_x_file, train_y_file, image_size, num_images=60000)
    test_images, test_labels = read_gz_files(dir, test_x_file, test_y_file, image_size, num_images=10000)

    val_ratio = 0.1
    val_idxs = np.random.choice(len(train_images), size=int(val_ratio * len(train_images)), replace=False)

    val_images = train_images[val_idxs]
    val_labels = train_labels[val_idxs]

    assert val_images.shape == (6000, 28, 28)
    assert val_labels.shape == (6000, )

    train_idxs = list(set(np.arange(len(train_images))) - set(val_idxs))
    train_images = train_images[train_idxs]
    train_labels = train_labels[train_idxs]

    assert len(set(train_idxs).intersection(set(val_idxs))) == 0

    original_data = []
    original_data.append(train_images)
    original_data.append(val_images)
    original_data.append(test_images)
    original_data.append(train_labels)
    original_data.append(val_labels)
    original_data.append(test_labels)
    save_as = f"original_{args.dataset.lower()}.npz"

    np.savez(os.path.join(dir, save_as), 
                train_images=original_data[0],
                val_images=original_data[1],
                test_images=original_data[2],
                train_labels=original_data[3],
                val_labels=original_data[4],
                test_labels=original_data[5])

    decoyed_data = []

    decoyed_images_train, decoy_positions_train = augment(train_images, train_labels, number_of_labels, image_size, patch_size, randomise=False, targets=args.targets)    
    decoyed_data.append(decoyed_images_train)

    decoyed_images_val, decoy_positions_val = augment(val_images, val_labels, number_of_labels, image_size, patch_size, randomise=True, targets=None)
    decoyed_data.append(decoyed_images_val)

    decoyed_images_test, decoy_positions_test = augment(test_images, test_labels, number_of_labels, image_size, patch_size, randomise=True, targets=None)
    decoyed_data.append(decoyed_images_test)

    decoyed_images_test_cor, decoy_positions_test_cor = augment(test_images, test_labels, number_of_labels, image_size, patch_size, randomise=False, targets=None)
    decoyed_data.append(decoyed_images_test_cor)

    decoyed_data.append(train_labels)
    decoyed_data.append(val_labels)
    decoyed_data.append(test_labels)
    decoyed_data.append(test_labels)

    decoyed_data.append(decoy_positions_train)
    decoyed_data.append(decoy_positions_val)
    decoyed_data.append(decoy_positions_test)
    decoyed_data.append(decoy_positions_test_cor)

    save_as = f"decoyed_{args.dataset.lower()}"
    if args.targets is not None:
        save_as = save_as + '_'
        for target in args.targets:
            save_as = save_as + f'{target}'
    save_as = save_as + '.npz'

    np.savez(os.path.join(dir, save_as), 
                train_images=decoyed_data[0],
                val_images=decoyed_data[1],
                test_images=decoyed_data[2],
                test_cor_images=decoyed_data[3],
                train_labels=decoyed_data[4],
                val_labels=decoyed_data[5],
                test_labels=decoyed_data[6],
                test_cor_labels=decoyed_data[7],
                decoy_positions_train=decoyed_data[8],
                decoy_positions_val=decoyed_data[9],
                decoy_positions_test=decoyed_data[10],
                decoy_positions_test_cor=decoyed_data[11])

if __name__ == "__main__":
    args = parse_args()
    main(args)