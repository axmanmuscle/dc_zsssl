# we need stuff like a data class, etc
# need to decide do we load the data first (like from disk) and use this class for a single image? 
# i think that's what we want.

import torch

class ssl_dataset(torch.utils.data.Dataset):
    """
    this class is going to assume that you've loaded the k-space samples
    from the image you want to reconstruct (e.g. fastmri)
    
    this is now a data loader over those samples with a given mask
    """

    def __init__(self, kspace_samples, train_mask):
        self.train_kspace = kspace_samples
        self.train_mask = train_mask

    def __len__(self):
        return len(self.train_kspace)

    def __getitem__(self, idx):
        sample = self.train_kspace(idx)
        return sample
