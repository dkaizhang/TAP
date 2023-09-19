import numpy as np
import os
import pandas as pd
import random
import shutil

from matplotlib import font_manager
from pathlib import Path
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont

def split_val(train_path, val_path, case, val_split):
    case_path = Path(os.path.join(train_path, case))
    case_images = [*case_path.glob('*.jpeg')]
    print(f'{case} images', len(case_images))

    val_case_images = random.sample(case_images, k=int(val_split * len(case_images)))
    for img in val_case_images:
        shutil.move(Path(img), os.path.join(val_path, case,os.path.basename(img)))
    len_train_case = len([*case_path.glob('*.jpeg')])
    len_val_case =len([*Path(os.path.join(val_path,case)).glob('*.jpeg')])
    print(f'train {case}', len_train_case)
    print(f'val {case}', len_val_case)
    print(f'total {case}', len_train_case + len_val_case)

def generate_labels(path):    
    normal_path = Path(os.path.join(path, 'NORMAL'))
    normal_images = [*normal_path.glob('*.jpeg')]
    normal_labels = [0 for i in range(len(normal_images))]
    print(f"found {len(normal_labels)} normal images")

    normal = pd.DataFrame({'img_path':normal_images, 'label':normal_labels})

    pneu_path = Path(os.path.join(path, 'PNEUMONIA'))    
    pneu_images = [*pneu_path.glob('*.jpeg')]
    pneu_labels = [1 for i in range(len(pneu_images))]
    print(f"found {len(pneu_labels)} pneumonia images")

    pneu = pd.DataFrame({'img_path':pneu_images, 'label':pneu_labels})

    print(f"found {len(pneu_labels) + len(normal_labels)} images")

    combined = pd.concat([normal, pneu], ignore_index=True)
    combined = combined.sample(frac=1)
    combined.to_csv(os.path.join(path, f"{os.path.basename(path)}_data.csv"))

    return len(pneu_labels) + len(normal_labels)

def crop_and_resize(path, target_size):
    img_paths = [*Path(path).glob('**/*.jpeg')]
    for img_path in img_paths:
        img = Image.open(img_path)

        max_len = min(img.size)
        left = (img.size[0] - max_len) // 2
        top = (img.size[1] - max_len) // 2
        right = (img.size[0] + max_len) // 2
        bottom = (img.size[1] + max_len) // 2        

        img = img.crop((left, top, right, bottom))
        img = img.resize((target_size, target_size))

        img = img.convert('L') if img.mode == 'RGB' else img

        img.save(img_path)

def text_artifact(decoy_path, data_csv, artifacts, mode, number_of_labels=2, downscale_by=10, targets=None):
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
        if mode == 'train' or mode == 'test_cor':
            if targets is None or label in set(targets):
                text = artifacts[label]
            else:
                text = ""
                decoy_present = False
        else:
            # text = artifacts[random.randint(0, number_of_labels - 1)]
            text = artifacts[1 - label]
            # text = artifacts[label]
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

        new_img_path = f'{decoy_path}/images/{os.path.basename(file)}'
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
    new_df.to_csv(f'{decoy_path}/{mode}_data.csv')

    if mode == 'train':
        decoy_positions = np.asarray(decoy_positions)
        np.save(f'{decoy_path}/decoy_positions.npy', decoy_positions)
    if mode == 'test':
        decoy_positions = np.asarray(decoy_positions)
        np.save(f'{decoy_path}/decoy_positions_test.npy', decoy_positions)

def stripe_artifact(decoy_path, data_csv, mode, number_of_labels=2, downscale_by=40, targets=None):
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
            value = 255 - (1-label) * step
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

        new_img_path = f'{decoy_path}/images/{os.path.basename(file)}'
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
    new_df.to_csv(f'{decoy_path}/{mode}_data.csv')

    if mode == 'train':
        decoy_positions = np.asarray(decoy_positions)
        np.save(f'{decoy_path}/decoy_positions.npy', decoy_positions)

if __name__ == '__main__':
    train_path = 'data/pneu/chest_xray/train'
    val_split = 0.1
    val_path = 'data/pneu/chest_xray/val'
    test_path = 'data/pneu/chest_xray/test'
    seed = 0
    target_size = 224
    crop_resize = True

    random.seed(seed)

    if not os.path.exists(val_path):
        os.makedirs(os.path.join(val_path,'NORMAL'))
        os.makedirs(os.path.join(val_path,'PNEUMONIA'))

        case = 'NORMAL'
        split_val(train_path, val_path, case, val_split)

        case = 'PNEUMONIA'
        split_val(train_path, val_path, case, val_split)

    len_train = generate_labels(train_path)
    len_val = generate_labels(val_path)
    len_test = generate_labels(test_path)
    print(len_train+len_val+len_test)

    if crop_resize:
        crop_and_resize(train_path, target_size)
        crop_and_resize(val_path, target_size)
        crop_and_resize(test_path, target_size)

    if True:
        print('text')
        # text artifacts
        artifacts = ['ABC', 'XYZ']

        train_text_path = 'data/pneu/chest_xray/train_text'
        val_text_path = 'data/pneu/chest_xray/val_text'
        test_text_path = 'data/pneu/chest_xray/test_text'
        test_text_cor_path = 'data/pneu/chest_xray/test_text_cor'

        if not os.path.exists(train_text_path):
            print('making train')
            os.makedirs(os.path.join(train_text_path,'images'))
            text_artifact(decoy_path=train_text_path, data_csv='data/pneu/chest_xray/train/train_data.csv', artifacts=artifacts, mode='train')
        if not os.path.exists(val_text_path):
            print('making val')
            os.makedirs(os.path.join(val_text_path,'images'))
            text_artifact(decoy_path=val_text_path, data_csv='data/pneu/chest_xray/val/val_data.csv', artifacts=artifacts, mode='val')
        if not os.path.exists(test_text_path):
            print('making test')
            os.makedirs(os.path.join(test_text_path,'images'))
            text_artifact(decoy_path=test_text_path, data_csv='data/pneu/chest_xray/test/test_data.csv',artifacts=artifacts, mode='test')
        if not os.path.exists(test_text_cor_path):
            print('making test_cor')
            os.makedirs(os.path.join(test_text_cor_path,'images'))
            text_artifact(decoy_path=test_text_cor_path, data_csv='data/pneu/chest_xray/test/test_data.csv',artifacts=artifacts, mode='test_cor')

    # stripe artifacts
    if True:
        print('stripe')
        train_stripe_path = 'data/pneu/chest_xray/train_stripe'
        val_stripe_path = 'data/pneu/chest_xray/val_stripe'
        test_stripe_path = 'data/pneu/chest_xray/test_stripe'
        test_stripe_cor_path = 'data/pneu/chest_xray/test_stripe_cor'

        if not os.path.exists(train_stripe_path):
            print('making train')
            os.makedirs(os.path.join(train_stripe_path,'images'))
            stripe_artifact(decoy_path=train_stripe_path, data_csv='data/pneu/chest_xray/train/train_data.csv', mode='train')
        if not os.path.exists(val_stripe_path):
            print('making val')
            os.makedirs(os.path.join(val_stripe_path,'images'))
            stripe_artifact(decoy_path=val_stripe_path, data_csv='data/pneu/chest_xray/val/val_data.csv', mode='val')
        if not os.path.exists(test_stripe_path):
            print('making test')
            os.makedirs(os.path.join(test_stripe_path,'images'))
            stripe_artifact(decoy_path=test_stripe_path, data_csv='data/pneu/chest_xray/test/test_data.csv', mode='test')
        if not os.path.exists(test_stripe_cor_path):
            print('making test_cor')
            os.makedirs(os.path.join(test_stripe_cor_path,'images'))
            stripe_artifact(decoy_path=test_stripe_cor_path, data_csv='data/pneu/chest_xray/test/test_data.csv', mode='test_cor')


