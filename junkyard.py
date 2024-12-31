# junkyard code
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

def kspace_to_imspace(kspace):

    im_space = np.fft.ifftshift( np.fft.ifftn( np.fft.fftshift( kspace, axes=(0, 1)),  axes=(0, 1) ), axes=(0,1))

    return im_space

def view_im(kspace, title=''):

    im_space = kspace_to_imspace(kspace)

    plt.imshow( np.abs( im_space ), cmap='grey')

    if len(title) > 0:
        plt.title(title)
    plt.show()

def main():

    data_dir = '/Volumes/T7 Shield/FastMRI/knee/singlecoil_train'
    fnames = glob.glob(data_dir +'/*')

    print(len(fnames))
    fname = fnames[25]
    slice_num = 28

    print(fname)
    with h5py.File(fname, 'r') as hf:
        ks = hf['kspace'][slice_num]

    view_im(ks)
    
    return 0

if __name__ == "__main__":
    main()


# def train(data_loader, model, loss_fun, optimizer, device = torch.device('cpu')):
#   """
#   run one training step

#   junkyard for now
#   """

#   for idx, data in enumerate(data_loader):
#     data = data.to(device)

#     out = model(data) # inputs on this line will depend on how the model is set up
#     # this is an interesting point because "model" will need to encompass the fourier transforms
#     train_loss = loss_fun(out, gt) # gt needs to come from the data loader?
