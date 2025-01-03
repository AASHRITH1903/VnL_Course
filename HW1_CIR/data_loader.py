import torch.utils.data as data
import os
import json
import torch


class Dataset(data.Dataset):
    def __init__(self, root, data_file_name, return_target=True):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)

        self.ids = range(len(self.data))
        self.return_target = return_target

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        id = self.ids[index]

        candidate_asin = data[id]['candidate']
        candidate_img_name = candidate_asin + '.png'
        if self.return_target:
            target_asin = data[id]['target']
            target_img_name = target_asin + '.png'
        else:
            target_asin = ''

        caption_texts = data[id]['captions']

        return {'target': target_asin, 'candidate': candidate_asin, 'caption': caption_texts}, os.path.join(self.root, candidate_img_name), os.path.join(self.root, target_img_name), ". ".join(caption_texts)

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    meta, ref_image_paths, tgt_image_paths, raw_captions = zip(*data)

    return meta, ref_image_paths, tgt_image_paths, raw_captions

def get_loader(root, data_file_name, batch_size, shuffle, return_target, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # relative caption dataset
    dataset = Dataset(root=root,
                      data_file_name=data_file_name,
                      return_target=return_target)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              timeout=60)

    return data_loader
