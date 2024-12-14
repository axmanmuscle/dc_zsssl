"""
zero shot self supervised learning
"""
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np

def train(data_loader, model, loss_fun, optimizer, device = torch.device('cpu')):
  """
  run one training step
  """

  for idx, data in enumerate(data_loader):
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

def make_masks(n):
  """
  i don't actually know what I should do here
  maybe a random subset of columns?
  """

  return np.zeros(n)


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

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  return 0

if __name__ == "__main__":
  main()