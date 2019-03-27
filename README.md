# EDSR_GAN
### EDSR implementation with GAN

This project is a test study of SISR models with GAN (most concretly EDSR).

I followed [the original paper of EDSR](https://arxiv.org/abs/1707.02921) , alongside with 
the original implementation of [SRGAN](https://arxiv.org/abs/1609.04802) for the adversarial part.


Here is the list of available options

```
-p for the patch size
-s for the scale factor
-l for the type of loss (choose from ['GAN','MSE','VGG','GAN,MSE','VGG,GAN','VGG,MSE'])
-e for the number of epochs
-v for the version of the model (to better manage different parameters)
--load if you desire to load the most recent checkpoint of the choosen version
-r the residual network length
-lr for the learning rate
--cuda for which GPU to use (no multi GPU available at the moment)
-k for the gan_k (how many times we should train the discriminator per batch)
-o the size of the classifier
--depth the depth of the discriminator
--batch the batch size
--eval_epoch if you desire to eval a given epoch of the current version and save the outputs
-eval if you want to do the evaluation
``` 

To train you just need to 

```
python main.py [desired options]
```

It will automatically create log files on a out directory that will be created if it doesn't exist.
You can plot the results using the functions in ``outils``
