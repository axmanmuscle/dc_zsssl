# To Do

For now, I think the right thing to do is make an "epoch" just a single pass of 
 - IFFT k-space
 - apply unet
 - fft k-space

and compare the loss against the training points?

This would keep the training "mask" fixed and remove having to use a data loader i believe

Okay so we have a way to generate a mask. The way forward is:

 - read in data file
 - apply mask (this is now our "undersampled k-space")
 - select 85% of the samples as train and the remaining 15% as validation
 - for i in epochs:
   - take the training data and apply the model
   - calculate the loss
   - update parameters
   - calculate validation loss
   - break is num epochs reached or validation loss doesnt change in long enough

Seems okay?

## 12/20
Need to split complex -> float x2 for the FFT and then backwards. at the unet it needs to be float x2

## 12/30
Need to write it so the sizes are consistent through the FFT and the unet. seems like the unet is having some issues

