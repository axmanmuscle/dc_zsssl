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

def training_loop(training_data, val_data, all_training_data, loss_mask, val_mask, model, loss_fun, optimizer, num_epochs = 50, device = torch.device('cpu')):
  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """

  model.train()

  avg_train_loss = 0
  training_data = training_data.to(device)
  all_training_data = all_training_data.to(device)
  val_data = val_data.to(device)

  tl_ar = []
  vl_ar = []

  for idx in range(num_epochs):
    out = model(training_data) # inputs on this line will depend on how the model is set up
    # this is an interesting point because "model" will need to encompass the fourier transforms
    train_loss = loss_fun(out, all_training_data, loss_mask) # gt needs to come from the data loader?
    val_out = model(all_training_data)
    val_loss = loss_fun(val_out, val_data, val_mask)

    tl_ar.append(train_loss.detach().numpy())
    vl_ar.append(val_loss.detach().numpy())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    print(f'on step {idx} of {num_epochs} with tl {round(float(train_loss.data), 3)} and vl {round(float(val_loss.data), 3)}')

  plt.plot(tl_ar)
  plt.plot(vl_ar)
  plt.legend(['training loss', 'val loss'])
  plt.show()

  out = model(training_data)
  view_im(np.squeeze(out.detach().numpy()))

def make_masks(sImg, rng, samp_frac, train_frac, loss_frac = 0.3):
  """
  i don't actually know what I should do here
  maybe a random subset of columns?

  yeah lets do a subset of columns and some amount chosen in the middle
  take the 10% of middle columns and some fraction of the rest

  INPUTS:
    sImg - dimensions of image
    rng - numpy random number generator
    samp_frac - fraction (0 < x < 1) of columns to use as undersampling
    train_frac - fraction (9 < x < 1) of samples to use as training, > 0.85 recommended
    loss_frac - fraction (0 < x < 1) of training mask to use as the loss calculation, < 0.3 recommended

  """

  # make sampling mask

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

  # make training mask

  train_num = round(train_frac * num_samples)
  train_chosen = rng.choice(num_samples, train_num, replace=False)

  train_mask = np.zeros(sImg)
  for idx in range(train_num):
    mask_idx = train_chosen[idx]
    train_mask[mask_rows[mask_idx], mask_cols[mask_idx]] = 1

  train_mask_indices = np.where(train_mask == 1)
  num_train_samples = len(train_mask_indices[0])

  train_mask_rows = train_mask_indices[0]
  train_mask_cols = train_mask_indices[1]
  # make loss mask??
  # this is a subset of the training mask

  loss_num = round(loss_frac * train_num)
  loss_chosen = rng.choice(train_num, loss_num, replace=False)

  loss_mask = np.zeros(sImg)
  for idx in range(loss_num):
    mask_idx = loss_chosen[idx]
    loss_mask[train_mask_rows[mask_idx], train_mask_cols[mask_idx]] = 1

  rest_training_mask = train_mask - loss_mask 
  val_mask = mask - train_mask

  # leftover_num_training = np.where(rest_training_mask == 1)

  # print(f'num samples in total image: {sImg[0]} * {sImg[1]} = {np.prod(sImg)}')
  # print(f'num samples in undersampled image: {num_samples}')
  # print(f'num samples in training mask: {train_num}')
  # print(f'num samples in training mask from mask: {num_train_samples}')
  # print(f'num samples in loss mask: {loss_num}')
  # print(f'num samples leftover for training: {len(leftover_num_training[0])}')
  # print(f'num samples for validation: {np.sum(val_mask)}')

  # plt.imshow(train_mask, cmap='gray')
  # plt.title('training mask')

  # plt.show()

  # plt.imshow(loss_mask, cmap='gray')
  # plt.title('loss mask')

  # plt.show()

  # plt.imshow(rest_training_mask, cmap='gray')
  # plt.title('leftover training')

  # plt.show()

  # plt.imshow(val_mask, cmap='gray')
  # plt.title('validation mask')
  
  # plt.show()

  return mask.astype(np.float32), train_mask.astype(np.float32), loss_mask.astype(np.float32) # convert to 32bit float to prevent uptyping

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
  slice_num = 18
  with h5py.File(fnames[file_num], 'r') as hf:
    ks = hf['kspace'][slice_num]
  
  mval = np.max(np.abs(ks))
  ks /= mval
  samp_frac = 0.4
  train_frac = 0.85
  rng = np.random.default_rng(seed=12202024)
  torch.manual_seed(12202024)
  sImg = ks.shape

  data_mask, training_mask, loss_mask = make_masks(sImg, rng, samp_frac, train_frac)

  train_split = training_mask - loss_mask

  sub_kspace = data_mask * ks
  all_training_kspace = training_mask * ks
  training_kspace = train_split * ks
  val_mask = data_mask - training_mask
  val_kspace = val_mask * ks

  sub_kspace = torch.tensor(sub_kspace)
  all_training_kspace = torch.tensor(all_training_kspace)
  training_kspace = torch.tensor(training_kspace)
  val_kspace = torch.tensor(val_kspace)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # view_im(sub_kspace, 'undersampled k-space')
  # view_im(ks, 'fully sampled image')

  # loss_fn = lambda x, y: np.linalg.norm(x - y, 'fro')
  # loss_fn = nn.MSELoss()

  model = zs_model()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  all_training_kspace = all_training_kspace[None, None, :, :]
  training_kspace = training_kspace[None, None, :, :]
  val_kspace = val_kspace[None, None, :, :]

  loss_mask = torch.tensor(loss_mask)
  val_mask = torch.tensor(val_mask)

  # plt.imshow(loss_mask, cmap='gray')
  # plt.show()
  training_loop(training_kspace, val_kspace, all_training_kspace, loss_mask, val_mask, model, math_utils.mixed_loss, optimizer, 40, device)


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