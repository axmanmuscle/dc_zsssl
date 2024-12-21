# To Do

For now, I think the right thing to do is make an "epoch" just a single pass of 
 - IFFT k-space
 - apply unet
 - fft k-space

and compare the loss against the training points?

This would keep the training "mask" fixed and remove having to use a data loader i believe

