import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import cPickle as pickle
import numpy as np
import nltk
from PIL import Image
import re
from build_vocab import Vocab



class AIDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, img_dir, caption_file, vocab, transform=None):
        '''
        Args:
            img_dir: Direcutory with all the images
            caption_file: Path to the factual caption file
            vocab: Vocab instance
            transform: Optional transform to be applied
        '''
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.vocab = vocab
        self.transform = transform

    def _get_imgname_and_caption(self, caption_file):
        '''extract image name and caption from factual caption file'''
        with open(caption_file, 'r') as f:
            res = f.readlines()

        imgname_caption_list = []

        for line in res:
            img_and_cap = line.strip().split(' ',1)
            # print img_and_cap
            img_and_cap = [x for x in img_and_cap]
            imgname_caption_list.append(img_and_cap)

        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list)

    def __getitem__(self, ix):
        '''return one data pair (image and captioin)'''
        img_name = self.imgname_caption_list[ix][0]
        img_name = img_name.split('#gxr#')[0]+'.jpg'
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list[ix][1]

        #######################################################

        image = Image.open(os.path.join(img_name))


        if self.transform is not None:

            image = self.transform(image)
        ######################################################
        tokens = nltk.tokenize.word_tokenize(caption)
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))

        target = torch.Tensor(caption)

        return image, target



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths





def get_loader(img_dir, caption_file, vocab,  batch_size, transform=None, shuffle=False, num_workers=0):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    AI = AIDataset(img_dir=img_dir,
                     caption_file=caption_file,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=AI,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader







# class AIDataset(data.Dataset):
#     '''ATset pretrain model decode Weibo dataset'''
#
#     def __init__(self, img_dir, caption_file, vocab, transform=None):
#         '''
#         Args:
#             img_dir: Direcutory with all the images
#             caption_file: Path to the factual caption file
#             vocab: Vocab instance
#             transform: Optional transform to be applied
#         '''
#         self.img_dir = img_dir
#         self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
#         self.vocab = vocab
#         self.transform = transform
#
#     def _get_imgname_and_caption(self, caption_file):
#         '''extract image name and caption from factual caption file'''
#         with open(caption_file, 'r') as f:
#             res = f.readlines()
#
#         imgname_caption_list = []
#         r = re.compile(' ',1)
#         for line in res:
#             img_and_cap = line.strip().split(' ',1)
#             # print img_and_cap
#             img_and_cap = [x for x in img_and_cap]
#             imgname_caption_list.append(img_and_cap)
#
#         return imgname_caption_list
#
#     def __len__(self):
#         return len(self.imgname_caption_list)
#
#     def __getitem__(self, ix):
#         '''return one data pair (image and captioin)'''
#         img_name = self.imgname_caption_list[ix][0]
#         img_name = img_name.split('#gxr#')[0]+'.jpg'
#         img_name = os.path.join(self.img_dir, img_name)
#         caption = self.imgname_caption_list[ix][1]
#
#         #######################################################
#         image = Image.open(img_name).convert('RGB')
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#         scaler = transforms.Resize((224, 224))
#         to_tensor = transforms.ToTensor()
#         image = normalize(to_tensor(scaler(image)))
#         ######################################################
#         tokens = nltk.tokenize.word_tokenize(caption)
#         caption = []
#         caption.append(self.vocab('<start>'))
#         caption.extend([self.vocab(token) for token in tokens])
#         caption.append(self.vocab('<end>'))
#
#         target = torch.Tensor(caption)
#
#         return image, target
#
#
# def collate_fn(data):
#     """Creates mini-batch tensors from the list of tuples (image, caption).
#
#     We should build custom collate_fn rather than using default collate_fn,
#     because merging caption (including padding) is not supported in default.
#
#     Args:
#         data: list of tuple (image, caption).
#             - image: torch tensor of shape (3, 256, 256).
#             - caption: torch tensor of shape (?); variable length.
#
#     Returns:
#         images: torch tensor of shape (batch_size, 3, 256, 256).
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length (descending order).
#     # data.sort(key=lambda x: len(x[1]), reverse=True)
#     # images, captions = zip(*data)
#     #
#     # # Merge images (from tuple of 3D tensor to 4D tensor).
#     # images = torch.stack(images, 0)
#     #
#     # # Merge captions (from tuple of 1D tensor to 2D tensor).
#     # lengths = [len(cap) for cap in captions]
#     # targets = torch.zeros(len(captions), max(lengths)).long()
#     # for i, cap in enumerate(captions):
#     #     end = lengths[i]
#     #     targets[i, :end] = cap[:end]
#
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     images, captions = zip(*data)
#
#     # images : tuple of 3D tensor -> 4D tensor
#     images = torch.stack(images, 0)
#
#     # captions : tuple of 1D Tensor -> 2D tensor
#     #lengths = torch.LongTensor([len(cap) for cap in captions])
#     lengths = [len(cap) for cap in captions]
#     captions = [pad_sequence(cap, max(lengths)) for cap in captions]
#     targets = torch.stack(captions, 0)
#
#
#     return images, targets, lengths
#
# def pad_sequence(seq, max_len):
#     seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
#     return seq
#
#
# def get_loader(root, caption_file, vocab, batch_size, transform=None, shuffle=False, num_workers=0):
#     """Returns torch.utils.data.DataLoader for custom coco dataset."""
#     # COCO caption dataset
#     AI = AIDataset(root, caption_file, vocab, transform)
#
#     # Data loader for COCO dataset
#     # This will return (images, captions, lengths) for every iteration.
#     # images: tensor of shape (batch_size, 3, 224, 224).
#     # captions: tensor of shape (batch_size, padded_length).
#     # lengths: list indicating valid length for each caption. length is (batch_size).
#     data_loader = torch.utils.data.DataLoader(dataset=AI,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=num_workers,
#                                               collate_fn=collate_fn)
#     return data_loader

if __name__ == "__main__":
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print vocab.i2w[10]
    img_path = "/home/gexuri/AIChallengeData/ai_challenger_caption_validation_20170910/caption_validation_images_20170910"
    cap_path = "/home/gexuri/VisualSearch/AIchallengetrain_val/TextData/seg.AIchallengetrain.caption.txt"

    data_loader = get_loader(img_path, cap_path, vocab, 1)
    print len(data_loader)
    # for i, (images, captions, lengths) in enumerate(data_loader):
    #     print(i)
    #     print(lengths)
    #     print(images)
    #     print(captions[:, 1:])
    #     print(captions)
    #     for wordindex in captions[0, 1:]:
    #         print vocab.i2w[wordindex]
    #
    #     # print()
    #     if i == 0:
    #         break