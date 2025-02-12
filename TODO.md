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

## 12/30 (2)
Need to add checkpointing to save the model. Need to add calculating validation loss and finding a good cutoff. Need to evaluate reconstruction accuracy.

## 1/3
Still need to add checkpointing.

More importantly - into the training method we need to hand the full set of training data and the validation data along with the loss mask. The training data is then split into the actual training data and the loss data.

A training step is then:
  - Take the actual training data (the given training data minus the loss data)
  - Run this through the model
  - Compute loss between this and the loss data at the loss mask
  - Run the model again on all of the training data (the previous training data and the loss data)
  - Compute loss between this and the validation data at the validation mask


## 1/15
Okay we need to redo how we do the training. Initially split the acquired data into training and validation. Then fix $k$ and split the training data into $k$ different training masks and loss masks. Then a single epoch trains over all of those $k$ training masks evaluated at the loss masks. Thus each epoch will be considerably more expensive but the authors claim this is vital.

Also we need to split some of this up. Put the masking stuff into utils for code readability.

## 1/17
Some stuff with the masks is working. Need to fix the loss calculation (with tensor.detach()) to properly calculate loss.

## 2/4
Looks like scaling was the real problem all along. Seems (?) to be working now. Run an experiment with smaller sample fractions to confirm. Try to get the UNet working with different sizes of images.

## 2/5
Making an unrolled network. IDK how to test - maybe start with applying it to a normal proximal grad descent problem?

