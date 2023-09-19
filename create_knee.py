import numpy as np
import os
import pandas as pd
import random

from matplotlib import font_manager
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_csv(base):

    splits = ['train', 'val', 'test']
    for split in splits:
        path = Path(os.path.join(base,split))
        files = [*path.glob('**/*.png')]
        print('files found: ',len(files))

        labels = [int(file.parent.stem) for file in files]
        df = pd.DataFrame({'img_path' : files, 'label' : labels})
        df.to_csv(os.path.join(base,f'{split}.csv'))

def text_artifact(base, decoy_path, data_csv, artifacts, mode, downscale_by=10, targets=None):
    decoy_positions = []

    df = pd.read_csv(data_csv)

    new_img_paths = []
    new_labels = []
    tag = []

    font = font_manager.FontProperties(family='monospace', weight='bold')
    font_file = font_manager.findfont(font)

    for i in range(len(df)):

        file = df.loc[i]['img_path']
        label = int(df.loc[i]['label'])

        decoy_present = True
        if mode == 'train' or mode =='test_cor':
            if targets is None or label in set(targets):
                text = artifacts[label]
            else:
                text = ""
                decoy_present = False
        else:
            text = artifacts[random.randint(0, len(artifacts) - 1)]

        tag.append(text)

        img = Image.open(file)
        font_size = min(img.size) // downscale_by # font size in pixels
        font = ImageFont.truetype(font_file, font_size)

        image_size = min(img.size)

        #     # flip location
        # if mode == 'train':
        #     offset_width = label * (image_size - 2*font_size)
        # else:
        #     offset_width = (1-label) * (image_size - 2*font_size)

        offset_width = random.randint(0,1) * (image_size - 2*font_size)
        offset_height = random.randint(0,1) * (image_size - font_size)
        # offset_height = 0 # artifacts only at the top 

        starting_coord = (offset_width, offset_height)

        draw = ImageDraw.Draw(img)
        draw.text(xy=starting_coord,text=text,fill=255,font=font)

        new_img_path = f'{decoy_path}/{os.path.basename(file)}'
        new_img_paths.append(new_img_path)
        new_labels.append(label)

        img.save(new_img_path)

        # next to save the decoy positions
        if mode == 'train' or mode == 'test':
            # this should be HxWxC
            decoy_position = np.zeros_like(np.asarray(img))
            assert decoy_position.shape[0] == img.size[1]

            if decoy_present:
                decoy_position[max(0,starting_coord[1]): min(starting_coord[1] + font_size, 224), max(0, starting_coord[0]): min(starting_coord[0] + 2*font_size, 224)] = 1
            decoy_position = Image.fromarray(decoy_position)
            decoy_position = np.asarray(decoy_position)

            decoy_positions.append(decoy_position)

    new_df = pd.DataFrame({'img_path' : new_img_paths, 'label' : new_labels, 'tag' : tag})
    new_df.to_csv(os.path.join(base,f'{mode}_decoyed.csv'))

    if mode == 'train':
        decoy_positions = np.asarray(decoy_positions)
        np.save(os.path.join(base,'decoy_positions.npy'), decoy_positions)
    if mode == 'test':
        decoy_positions = np.asarray(decoy_positions)
        np.save(os.path.join(base,'decoy_positions_test.npy'), decoy_positions)

def stripe_artifact(base, decoy_path, data_csv, mode, number_of_labels=5, downscale_by=40, targets=None):
    decoy_positions = []

    step = 255 // number_of_labels

    df = pd.read_csv(data_csv)

    new_img_paths = []
    new_labels = []
    tag = []

    for i in range(len(df)):

        file = df.loc[i]['img_path']
        label = int(df.loc[i]['label'])

        decoy_present = True
        if mode == 'train' or mode == 'test_cor':
            if targets is None or label in set(targets):
                value = 255 - label * step
            else:
                value = None
                decoy_present = False
                
        else:
            value = 255 - random.randint(0, number_of_labels - 1) * step
        tag.append(value)

        img = Image.open(file)

        if value is not None:
            image_size = min(img.size)
            artifact_width = image_size // downscale_by
            jitter = artifact_width // 2
            if jitter == 0:
                print('no jitter')
            if random.randint(0,9) < 5:
                offset_width = random.randint(0,5) * jitter 
            else:
                offset_width = image_size - artifact_width - random.randint(0,5) * jitter

            starting_coord = (offset_width, 0)
            end_coord = (offset_width + artifact_width, image_size)

            draw = ImageDraw.Draw(img)
            draw.rectangle(xy=[starting_coord, end_coord], fill=value)

        new_img_path = f'{decoy_path}/{os.path.basename(file)}'
        new_img_paths.append(new_img_path)
        new_labels.append(label)

        img.save(new_img_path)

        # next to save the decoy positions
        if mode == 'train':
            # this should be HxWxC
            decoy_position = np.zeros_like(np.asarray(img))
            assert decoy_position.shape[0] == img.size[1]

            if decoy_present:
                # starting_coord is a (x,y) tuple which is an offset from top-left
                # starting_coord[1] is 0 which is the top, starting_coord[0] is the left corner of the artifact
                decoy_position[starting_coord[1] : end_coord[1], starting_coord[0] : 1 + end_coord[0]] = 1
            decoy_position = Image.fromarray(decoy_position)
            decoy_position = np.asarray(decoy_position)

            decoy_positions.append(decoy_position)

    if mode == 'train':
        assert len(decoy_positions) == len(new_img_paths)
    assert len(tag) == len(new_labels)
    assert len(tag) == len(new_img_paths)

    new_df = pd.DataFrame({'img_path' : new_img_paths, 'label' : new_labels, 'tag' : tag})
    assert len(new_df) == len(df)
    new_df.to_csv(os.path.join(base,f'{mode}_stripe.csv'))

    if mode == 'train':
        decoy_positions = np.asarray(decoy_positions)
        np.save(os.path.join(base,'decoy_positions_stripe.npy'), decoy_positions)

if __name__ == '__main__':
    base = 'data/kneeKL224'
    create_csv(base)

    if True:
        print('text')
        artifacts = ['ABC', 'DEF', 'GHI', 'JKL', 'MNO']

        decoy_train = 'data/kneeKL224/train_decoy'
        decoy_val = 'data/kneeKL224/val_decoy'
        decoy_test = 'data/kneeKL224/test_decoy'
        decoy_test_cor = 'data/kneeKL224/test_decoy_cor'

        if not os.path.exists(decoy_train):
            print('making train')
            os.makedirs(decoy_train)
            text_artifact(base, decoy_train, os.path.join(base,'train.csv'),artifacts,'train')
        if not os.path.exists(decoy_val):
            print('making val')
            os.makedirs(decoy_val)
            text_artifact(base, decoy_val, os.path.join(base,'val.csv'),artifacts,'val')
        if not os.path.exists(decoy_test):
            print('making test')
            os.makedirs(decoy_test)
            text_artifact(base, decoy_test, os.path.join(base,'test.csv'),artifacts,'test')
        if not os.path.exists(decoy_test_cor):
            print('making test_cor')
            os.makedirs(decoy_test_cor)
            text_artifact(base, decoy_test_cor, os.path.join(base,'test.csv'),artifacts,'test_cor')

    if True:
        print('stripe')
        decoy_train = 'data/kneeKL224/train_stripe'
        decoy_val = 'data/kneeKL224/val_stripe'
        decoy_test = 'data/kneeKL224/test_stripe'
        decoy_test_cor = 'data/kneeKL224/test_stripe_cor'

        if not os.path.exists(decoy_train):
            print('making train')
            os.makedirs(decoy_train)
            stripe_artifact(base, decoy_train, os.path.join(base,'train.csv'),'train')
        if not os.path.exists(decoy_val):
            print('making val')
            os.makedirs(decoy_val)
            stripe_artifact(base, decoy_val, os.path.join(base,'val.csv'),'val')
        if not os.path.exists(decoy_test):
            print('making test')
            os.makedirs(decoy_test)
            stripe_artifact(base, decoy_test, os.path.join(base,'test.csv'),'test')
        if not os.path.exists(decoy_test_cor):
            print('making test_cor')
            os.makedirs(decoy_test_cor)
            stripe_artifact(base, decoy_test_cor, os.path.join(base,'test.csv'),'test_cor')