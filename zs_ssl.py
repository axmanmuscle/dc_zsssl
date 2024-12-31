"""
zero shot self supervised learning
"""
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from unet import zs_model
import math_utils
import glob
import h5py
import matplotlib.pyplot as plt
from junkyard import view_im

def train(data, model, loss_fun, optimizer, num_epochs = 50, device = torch.device('cpu')):
  """
  run one training step
  we aren't changing the data mask so there is no need for a dataloader?
  """
  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """

  model.train()

  avg_train_loss = 0
  data = data.to(device)

  tl_ar = []

  for idx in range(num_epochs):
    out = model(data) # inputs on this line will depend on how the model is set up
    # this is an interesting point because "model" will need to encompass the fourier transforms
    train_loss = loss_fun(out, data) # gt needs to come from the data loader?
    tl_ar.append(train_loss.detach().numpy())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    print(f'on step {idx} of {num_epochs} with tl {train_loss}')

  plt.plot(tl_ar)
  plt.show()

def make_masks(sImg, rng, samp_frac, train_frac):
  """
  i don't actually know what I should do here
  maybe a random subset of columns?

  yeah lets do a subset of columns and some amount chosen in the middle
  take the 10% of middle columns and some fraction of the rest

  INPUTS:
    sImg - dimensions of image
    rng - numpy random number generator
  """

  numCols = sImg[1]

  center = numCols // 2

  fivePer = round(0.05 * numCols)
  lowerB = center - fivePer
  upperB = center + fivePer

  colRange = [i for i in range(numCols) if i < lowerB or i > upperB]

  colsChosen = rng.choice(colRange, round(samp_frac*len(colRange)), replace=False) 

  mask = np.zeros(sImg)
  mask[:, colsChosen] = 1
  mask[:, lowerB:upperB] = 1

  mask_indices = np.where(mask == 1)
  num_samples = len(mask_indices[0])

  mask_rows = mask_indices[0]
  mask_cols = mask_indices[1]

  train_num = round(train_frac * num_samples)
  train_chosen = rng.choice(num_samples, train_num, replace=False)

  train_mask = np.zeros(sImg)
  for idx in range(train_num):
    mask_idx = train_chosen[idx]
    train_mask[mask_rows[mask_idx], mask_cols[mask_idx]] = 1
  

  return mask.astype(np.float32), train_mask.astype(np.float32) # convert to 32bit float to prevent uptyping

def main():
  """
  here's the steps
    - load data
        - this will be an nxm matrix of (0-filled) k-space?
    - partition into train and validate indices
        - probably do this randomly
    - train the u-net
        - this process takes the IFT of the 0-filled training k-space samples, applies the unet, takes the FT of the output, compares to the data points
          - all of this belongs in the "model"
        - at each step compute treaining and validation loss
        - when the validation loss starts going back up that's when we stop (?)
    - once this is trained reconstruct the image
  """

  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """

  # data_dir = '/Volumes/T7 Shield/FastMRI/knee/singlecoil_train'
  data_dir = '/Users/alex/Desktop/fastMRI/knee_singlecoil_train'
  fnames = glob.glob(data_dir +'/*')

  file_num = 1
  slice_num = 10
  with h5py.File(fnames[file_num], 'r') as hf:
    ks = hf['kspace'][slice_num]
  
  samp_frac = 0.4
  train_frac = 0.85
  rng = np.random.default_rng(seed=12202024)
  sImg = ks.shape

  data_mask, training_mask = make_masks(sImg, rng, samp_frac, train_frac)

  sub_kspace = data_mask * ks
  training_kspace = training_mask * ks
  val_mask = data_mask - training_mask
  val_kspace = val_mask * ks

  sub_kspace = torch.tensor(sub_kspace)
  training_kspace = torch.tensor(training_kspace)
  val_kspace = torch.tensor(val_kspace)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # loss_fn = lambda x, y: np.linalg.norm(x - y, 'fro')
  loss_fn = nn.MSELoss()

  model = zs_model()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  training_kspace = training_kspace[None, None, :, :]

  train(training_kspace, model, math_utils.complex_mse_loss, optimizer, 75, device)


  # view_im(ks)
  # view_im(sub_kspace)
  # view_im(training_kspace)
  # view_im(val_kspace)

  return 0

if __name__ == "__main__":
  # rng = np.random.default_rng(2024)
  # sImg = [640, 320]
  # m, tm = make_masks(sImg, rng, 0.4, 0.85)
  # plt.imshow(tm, cmap='grey')
  # plt.show()


  main()