# trytry-vae

Variational autoencoder (VAE) is an extension of autoencoder (AE). The novelty of VAE is that it constrains the latent space with normal distribution, N(0, 1), so that the latent space has following properties:
1. **Continuity**
Two close points in the latent space should not give two completely different contents once decoded
2. **Completeness**
For a chosen distribution, a point sampled from the latent space should give “meaningful” content once decoded

Here is the architecture of a normal AE model:
![AE.png](https://miro.medium.com/max/770/1*bY_ShNK6lBCQ3D9LYIfwJg@2x.png)

Here is the architecture of a normal VAE model:
![VAE.png](https://miro.medium.com/max/770/1*Q5dogodt3wzKKktE0v3dMQ@2x.png)

Following is the latent spaces that AE and VAE learns (left for AE. right for VAE)
![latent1.png](https://miro.medium.com/max/1100/1*9ouOKh2w-b3NNOVx4Mw9bg@2x.png)
![latent2.png](https://miro.medium.com/max/1100/1*83S0T8IEJyudR_I5rI9now@2x.png)

## Training
```bash
# Train a vae model on mnist dataset
$ python main.py --config config/mnist.yml

# Train a cvae model on mnist dataset
$ python main.py --config config/mnist_cvae.yml
```

## Tensorboard
![tensorboard.png](https://i.imgur.com/LiFWkdM.png)

## Result
- Latent space of VAE trained on mnist dataset
![vae-latent.png](https://i.imgur.com/VSIogUb.png)

- Latent space of CVAE trained on mnist dataset
![cvae-latent.png](https://i.imgur.com/plpFurk.png)

- Generate new images with VAE (Interpolation)
![vae-gen.png](https://i.imgur.com/pUWEcpq.png)

- Generate new images with CVAE (Given the class information and z information)
![cvae-gen.png](https://i.imgur.com/yTVjlGJ.png)

## Reference
- https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
- https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
- https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/

## Mathematical Derivation
