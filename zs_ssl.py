"""
zero shot self supervised learning
"""
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from unet import zs_model, dc_zs_model
import math_utils
import glob
import h5py
import matplotlib.pyplot as plt
from junkyard import view_im
import utils
from tqdm import tqdm
import os

def training_loop(training_data, val_data, val_mask, tl_masks, model, loss_fun, optimizer, num_epochs = 50, device = torch.device('cpu')):
  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """
  directory = os.getcwd()
  model.train()

  training_data = training_data.to(device)
  val_data = val_data.to(device)
  val_mask = val_mask.to(device)

  vl_min = 10000

  tl_ar = []
  vl_ar = []
  ep = 0
  val_loss_tracker = 0
  val_stop_training = 10

  model_fname = f"best_{len(tl_masks)}.pth"

  while ep < num_epochs and val_loss_tracker < val_stop_training:
    avg_train_loss = 0.0
    for jdx, tl_mask in tqdm(enumerate(tl_masks)):
      #print(f'subiter {jdx}')
      tmask = tl_mask[0]
      lmask = tl_mask[1]

      tmask = torch.tensor(tmask)
      lmask = torch.tensor(lmask)
      tmask = tmask.to(device)
      lmask = lmask.to(device)

      tdata = training_data * tmask
      out = model(tdata) # inputs on this line will depend on how the model is set up
      # this is an interesting point because "model" will need to encompass the fourier transforms
      #if jdx < 1:
      if True:
        train_loss = loss_fun(out, training_data, lmask) # gt needs to come from the data loader?
      else:
        train_loss += loss_fun(out, training_data, lmask)
      avg_train_loss += train_loss.cpu().data
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

    val_out = model(training_data)
    val_loss = loss_fun(val_out, val_data, val_mask)
    vl_data = val_loss.cpu().data

    tl_ar.append(avg_train_loss / (jdx+1))
    vl_ar.append(vl_data)

    checkpoint = {
      "epoch": ep,
      "val_loss_min": vl_data,
      "model_state": model.state_dict(),
      "optim_state": optimizer.state_dict()
    }

    if vl_data <= vl_min:
      vl_min = vl_data
      torch.save(checkpoint, os.path.join(directory, model_fname))
      val_loss_tracker = 0
    else:
      val_loss_tracker += 1

    ep += 1

    # optimizer.zero_grad()
    # train_loss.backward()
    # optimizer.step()

    print('on step {} of {} with tl {:.2E} and vl {:.2E}'.format(ep, num_epochs, float(tl_ar[-1]), float(val_loss.data)))
    # print(f'on step {idx} of {num_epochs} with tl {round(float(train_loss.data), 3)} and vl {round(float(val_loss.data), 3)}')

  plt.plot(tl_ar)
  plt.plot(vl_ar)
  plt.legend(['training loss', 'val loss'])
  plt.show()

  best_checkpoint = torch.load(os.path.join(directory, model_fname))
  model.load_state_dict(best_checkpoint['model_state'])
  all_data = training_data+val_data

  out = model(all_data)
  oc = out.cpu()
  view_im(np.squeeze(oc.detach().numpy()))

def training_loop_dc(training_data, val_data, val_mask, tl_masks, model, loss_fun, optimizer, num_epochs = 50, device = torch.device('cpu')):
  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """
  directory = os.getcwd()
  model.train()

  training_data = training_data.to(device)
  val_data = val_data.to(device)
  val_mask = val_mask.to(device)

  training_mask = torch.abs(training_data) > 0
  training_mask = torch.squeeze(training_mask)
  all_tdata_consistency = training_data[:, :, training_mask]

  all_data = training_data + val_data
  alldata_mask = torch.abs(all_data) > 0
  alldata_mask = torch.squeeze(alldata_mask)
  alldata_consistency = all_data[:, :, alldata_mask]

  vl_min = 10000

  tl_ar = []
  vl_ar = []
  ep = 0
  val_loss_tracker = 0
  val_stop_training = 15

  model_fname = f"dc_best_{len(tl_masks)}.pth"

  while ep < num_epochs and val_loss_tracker < val_stop_training:
    avg_train_loss = 0.0
    for jdx, tl_mask in tqdm(enumerate(tl_masks)):
      #print(f'subiter {jdx}')
      tmask = tl_mask[0]
      lmask = tl_mask[1]

      tmask = torch.tensor(tmask)
      lmask = torch.tensor(lmask)
      tmask = tmask.to(device)
      lmask = lmask.to(device)

      tdata = training_data * tmask
      tdata_consistency = tdata[:, :, tmask > 0]
      out = model(tdata, tmask, tdata_consistency) # inputs on this line will depend on how the model is set up
      # this is an interesting point because "model" will need to encompass the fourier transforms
      #if jdx < 1:
      if True:
        train_loss = loss_fun(out, training_data, lmask) # gt needs to come from the data loader?
      else:
        train_loss += loss_fun(out, training_data, lmask)
      avg_train_loss += train_loss.cpu().data
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

    
    val_out = model(training_data, training_mask, all_tdata_consistency)
    val_loss = loss_fun(val_out, val_data, val_mask)
    vl_data = val_loss.cpu().data

    tl_ar.append(avg_train_loss / (jdx+1))
    vl_ar.append(vl_data)

    checkpoint = {
      "epoch": ep,
      "val_loss_min": vl_data,
      "model_state": model.state_dict(),
      "optim_state": optimizer.state_dict()
    }

    if vl_data <= vl_min:
      vl_min = vl_data
      torch.save(checkpoint, os.path.join(directory, model_fname))
      val_loss_tracker = 0
    else:
      val_loss_tracker += 1

    ep += 1

    # optimizer.zero_grad()
    # train_loss.backward()
    # optimizer.step()

    print('on step {} of {} with tl {:.2E} and vl {:.2E}'.format(ep, num_epochs, float(tl_ar[-1]), float(val_loss.data)))
    # print(f'on step {idx} of {num_epochs} with tl {round(float(train_loss.data), 3)} and vl {round(float(val_loss.data), 3)}')

  plt.plot(tl_ar)
  plt.plot(vl_ar)
  plt.legend(['training loss', 'val loss'])
  plt.show()

  best_checkpoint = torch.load(os.path.join(directory, model_fname))
  model.load_state_dict(best_checkpoint['model_state'])

  out = model(all_data, alldata_mask, alldata_consistency)
  oc = out.cpu()
  view_im(np.squeeze(oc.detach().numpy()))


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
  # data_dir = '/Users/alex/Desktop/fastMRI/knee_singlecoil_train'
  data_dir = '/home/alex/Documents/research/mri/knee_singlecoil_train'
  fnames = glob.glob(data_dir +'/*')

  file_num = 1
  slice_num = 22
  left_idx = 18
  right_idx = 350
  # here probably pad out to 512?
  # no no pad inside the 
  with h5py.File(fnames[file_num], 'r') as hf:
    ks = hf['kspace'][slice_num]
    ks_mask = ks[:, left_idx:right_idx]
  
  mval = np.max(np.abs(ks))
  ks /= mval
  samp_frac = 0.05
  train_frac = 0.85 # fraction of all samples devoted to training
  train_loss_split_frac = 0.8 # fraction of training samples devoted to training vs. loss
  rng = np.random.default_rng(seed=12202024)
  torch.manual_seed(12202024)
  sMask = ks_mask.shape
  sImg = ks.shape

  ## refactor: here we'll make a function to subsample k-space, a function to split into training and validation
  ## and then somewhere generate a bunch of different training/loss masks!

  k = 100 # not an informed choice
  usMask = utils.undersample_kspace(sMask, rng, samp_frac)
  undersample_mask = np.zeros(sImg)
  undersample_mask[:, left_idx:right_idx] = usMask
  train_mask, val_mask = utils.mask_split(undersample_mask, rng, train_frac)

  sub_kspace = undersample_mask * ks
  training_kspace = train_mask * ks
  val_kspace = val_mask * ks

  view_im(sub_kspace)

  sub_kspace = torch.tensor(sub_kspace)
  training_kspace = torch.tensor(training_kspace)
  val_kspace = torch.tensor(val_kspace)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #device = torch.device('cpu')

  # view_im(sub_kspace, 'undersampled k-space')
  # view_im(ks, 'fully sampled image')

  # loss_fn = lambda x, y: np.linalg.norm(x - y, 'fro')
  # loss_fn = nn.MSELoss()

  # view_im(ks, 'fully sampled')
  # model = zs_model()
  # kten = torch.tensor(ks)
  # kten = kten[None, None, :, :]
  # os = model(kten)
  # os = np.squeeze(os.detach().numpy())
  # view_im(os, 'after model')
  # return 0

  # model = zs_model()
  # model = model.to(device)

  # npr = 0
  # for pr in model.parameters():
  #   npr += 1
  #   print(f'ndc: level {npr} with len {pr.size()}')
  model = dc_zs_model(*(ks.shape))
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  training_kspace = training_kspace[None, None, :, :]
  val_kspace = val_kspace[None, None, :, :]

  val_mask = torch.tensor(val_mask)

  # we actually want to create all k masks outside of the training loop because we want to create them once
  tl_masks = []
  for idx in range(k):
    tm, lm = utils.mask_split(train_mask, rng, train_loss_split_frac)
    tl_masks.append((tm, lm))
    

  # training_loop(training_kspace, val_kspace, val_mask, tl_masks, model, math_utils.mixed_loss, optimizer, 100, device)

 

  # npr = 0
  # for pr in model.parameters():
  #   npr += 1
  #   print(f'dc: level {npr} with len {pr.size()}')

  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  training_loop_dc(training_kspace, val_kspace, val_mask, tl_masks,\
     model, math_utils.mixed_loss, optimizer, 100, device)



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