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
import utils

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

    print('on step {} of {} with tl {:.2E} and vl {:.2E}'.format(idx+1, num_epochs, float(train_loss.data), float(val_loss.data)))
    # print(f'on step {idx} of {num_epochs} with tl {round(float(train_loss.data), 3)} and vl {round(float(val_loss.data), 3)}')

  plt.plot(tl_ar)
  plt.plot(vl_ar)
  plt.legend(['training loss', 'val loss'])
  plt.show()

  out = model(training_data)
  view_im(np.squeeze(out.detach().numpy()))

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

  ## refactor: here we'll make a function to subsample k-space, a function to split into training and validation
  ## and then somewhere generate a bunch of different training/loss masks!

  k = 20 # not an informed choice
  undersample_mask = utils.undersample_kspace(sImg, rng, samp_frac)
  train_mask, val_mask = utils.mask_split(undersample_mask, rng, train_frac)

  sub_kspace = undersample_mask * ks
  training_kspace = train_mask * ks
  val_kspace = val_mask * ks

  sub_kspace = torch.tensor(sub_kspace)
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
  training_loop(training_kspace, val_kspace, all_training_kspace, loss_mask, val_mask, model, math_utils.mixed_loss, optimizer, 20, device)


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