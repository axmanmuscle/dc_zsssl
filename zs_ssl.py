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

  for idx in range(num_epochs):
    data = data.to(device)

    out = model(data) # inputs on this line will depend on how the model is set up
    # this is an interesting point because "model" will need to encompass the fourier transforms
    train_loss = loss_fun(out, gt) # gt needs to come from the data loader?

def load_data(fname):
  """
  this should just load the data into a matrix
  I'm making this a function in case there's any preprocessing we want to do on the data
  """

  x = sio.loadmat(fname)

  return x

def make_masks(sImg, rng, sampFrac = 0.4):
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

  colsChosen = rng.choice(colRange, round(sampFrac*len(colRange))) 

  mask = np.zeros(sImg)
  mask[:, colsChosen] = 1


  return mask

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

  data_dir = '/Volumes/T7 Shield/FastMRI/knee/singlecoil_train'
  fnames = glob.glob(data_dir +'/*')

  slice_num = 10
  with h5py.File(fnames[10], 'r') as hf:
    ks = hf['kspace'][slice_num]
  

  rng = np.random.default_rng(seed=12202024)
  sImg = np.array([64, 64, 2])
  testImg = rng.random(sImg)
  testImgCmplx = math_utils.np_to_complex(testImg)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  network = zs_model()

  network(testImgCmplx)

  return 0

if __name__ == "__main__":
  rng = np.random.default_rng(2024)
  sImg = [640, 320]
  m = make_masks(sImg, rng, 0.4)
  plt.imshow(m, cmap='grey')
  plt.show()


  # main()