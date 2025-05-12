"""
new training script for zero shot
want the ability to loop over lots of parameters and let it run overnight
"""

import numpy as np
import glob
import math_utils
import torch
import os
import h5py
import utils
from unet import zs_model, dc_zs_model
from tqdm import tqdm
import matplotlib.pyplot as plt


def training_loop(training_data, val_data, val_mask, tl_masks,
                  model, loss_fun, optimizer, data_consistency, 
                  val_stop_training = 15, num_epochs = 50, device = torch.device('cpu'),
                  directory=os.getcwd()):
  """
  todo:
    - the data in is undersampled k-space. split into train and validation (oh this should be done earlier)
    - the training data is split out and run through the model. compare loss
  """
  model.train()

  training_data = training_data.to(device)
  val_data = val_data.to(device)
  val_mask = val_mask.to(device)

  if data_consistency:
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

  model_fname = f"best_{len(tl_masks)}.pth"
  if data_consistency:
    model_fname = "dc_"+model_fname

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
      if data_consistency:
        tdata_consistency = tdata[:, :, tmask > 0]
        out = model(tdata, tmask, tdata_consistency)
      else:
        out = model(tdata) 
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

    if data_consistency:
      val_out = model(training_data, training_mask, all_tdata_consistency)
    else:
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
    ## optional
    # save images at each epoch
    all_data = training_data+val_data
    if data_consistency:
      out = model(all_data, alldata_mask, alldata_consistency)
    else:
      out = model(all_data)
    tstr = f'output_epoch{ep}.png'

    out = out.cpu()
    oc = np.squeeze(out.detach().numpy())

    im = math_utils.kspace_to_imspace(oc)
    img_dir = os.path.join(directory, 'imgs/')

    if not os.path.isdir(img_dir):
      os.mkdir(img_dir)
    plt.imsave(os.path.join(img_dir, tstr), np.abs( im ), cmap='grey')

  plt.plot(tl_ar)
  plt.plot(vl_ar)
  plt.legend(['training loss', 'val loss'])
  if data_consistency:
    loss_str = 'dc_loss_fig.png'
  else:
    loss_str = 'nodc_loss_fig.png'
  plt.savefig(os.path.join(directory, loss_str))

  best_checkpoint = torch.load(os.path.join(directory, model_fname))
  model.load_state_dict(best_checkpoint['model_state'])
  all_data = training_data+val_data

  if data_consistency:
    out = model(all_data, alldata_mask, alldata_consistency)
    tstr = 'dc_output.png'
  else:
    out = model(all_data)
    tstr = 'nodc_output.png'
  out = out.cpu()

  oc = np.squeeze(out.detach().numpy())

  im = math_utils.kspace_to_imspace(oc)
  # plt.imshow( np.abs( im ), cmap='grey')
  plt.imsave(os.path.join(directory, tstr), np.abs( im ), cmap='grey')

  plt.clf()

def run_training(ks, sImg, sMask, left_idx, right_idx, rng, samp_frac, train_frac, 
                 train_loss_split_frac, k, dc, results_dir,
                 val_stop_training, num_epochs=100):
                   
  usMask = utils.undersample_kspace_gaussian(sMask, rng, samp_frac)
  undersample_mask = np.zeros(sImg)
  undersample_mask[:, left_idx:right_idx] = usMask
  train_mask, val_mask = utils.mask_split(undersample_mask, rng, train_frac)

  sub_kspace = undersample_mask * ks
  training_kspace = train_mask * ks
  val_kspace = val_mask * ks

  sub_kspace = torch.tensor(sub_kspace)
  training_kspace = torch.tensor(training_kspace)
  val_kspace = torch.tensor(val_kspace)
  val_mask = torch.tensor(val_mask)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if dc:
    model = dc_zs_model(*(ks.shape))
  else:
    model = zs_model(*(ks.shape))
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  training_kspace = training_kspace[None, None, :, :]
  val_kspace = val_kspace[None, None, :, :]

  tl_masks = []
  for idx in range(k):
    tm, lm = utils.mask_split(train_mask, rng, train_loss_split_frac)
    tl_masks.append((tm, lm))

  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

  directory = f'sf{int(samp_frac*100)}p_tf{int(train_frac*100)}p_k{k}_vst{val_stop_training}'
  if dc:
    directory = "dc_" + directory
  directory = os.path.join(results_dir, directory)

  if not os.path.isdir(directory):
    os.mkdir(directory)

  training_loop(training_kspace, val_kspace, val_mask, tl_masks,
              model, math_utils.mixed_loss, optimizer, dc, 
              val_stop_training, num_epochs, device,
              directory)

def main():
  """
  read in data and decide what to iterate over
  """
  data_dir = '/home/alex/Documents/research/mri/knee_singlecoil_train'
  results_dir = '/home/alex/Documents/research/mri/results/brain_unet'
  fnames = glob.glob(data_dir +'/*')

  # file_num = 1
  # slice_num = 22
  # here probably pad out to 512?
  # no no pad inside the 
  # with h5py.File(fnames[file_num], 'r') as hf:
  #   ks = hf['kspace'][slice_num]
  #   w = np.where(np.abs(ks[0, :]) > 0)
  #   w = w[0]
  #   left_idx = w[0]
  #   right_idx = w[-1] + 1
  #   ks_mask = ks[:, left_idx:right_idx]

  import scipy.io as sio
  # data = sio.loadmat('/home/alex/Documents/research/mri/data/ankle_unet_data.mat')
  # ks = data['ks']

  data = sio.loadmat('/home/alex/Documents/research/mri/data/brain_p74240_slice76.mat')
  ks = data['f_ssq']
  
  
  mval = np.max(np.abs(ks))
  ks /= mval
  ks = ks.astype('complex64')
  rng = np.random.default_rng(seed=12202024)
  torch.manual_seed(12202024)
  # sMask = ks_mask.shape
  sMask = ks.shape
  sImg = ks.shape
  left_idx = 0
  right_idx = sImg[1]

  samp_fracs = [0.25]
  train_fracs = [0.95]
  train_loss_split_frac = 0.9
  k_s = [100]
  dcs = [True, False]
  val_stop_trainings = [25]

  for sf in samp_fracs:
    for tf in train_fracs:
      for k in k_s:
        for vst in val_stop_trainings:
          for dc in dcs:

            run_training(ks, sImg, sMask, left_idx, right_idx, rng, 
                      sf, tf, train_loss_split_frac, 
                      k, dc, results_dir, vst, 150)
  return 0
  
if __name__ == "__main__":
  main()