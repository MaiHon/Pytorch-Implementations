# _*Pytorch Implementations*_

[Code Reference](https://github.com/eriklindernoren/PyTorch-GAN)   I've referred to this page a lot.


Implementing Serveral Networks in [**pytorch**](https://pytorch.org) with studying it each papers. Welcome any advice with widely open arms.

<br></br>
<br></br>

[**Implementations**](#Implementations)
 + [GANs](#Gans)
   + [VanilaGAN](#vanilagan)
   + [DCGAN](#dcgan)

<br></br>

#### VanilaGAN
- Generative Adversarial Network
- Authors 
  - [Ian J. Goodfellow | Jean Pouget-Abadie | Mehdi Mirza | Bing Xu | David Warde-Farley | Sherjil Ozair | Aaron Courville | Yoshua Bengio]
- [[**Paper**]](https://arxiv.org/abs/1406.2661) | [[**Code**]](./Implementations/GANs/VanilaGAN/VanilaGAN.ipynb)
<p align="center">
    <img src='./gifs/VanilaGAN.gif' width="360"\>
</p>

#### DCGAN
- Deep Convolutional Generative Adversarial Networks
- Authors 
  - [Alec Radford | Luke Metz | Soumith Chintala]
<p align="center">
    <img src="./imgs/DCGAN.png" width="400"\>
</p>

- The Main Idea is embedding CNN into the VanilaGAN
- [[**Paper**]](https://arxiv.org/abs/1511.06434) | [[**Code**]](./Implementations/GANs/DCGAN/DCGAN.ipynb)
<p align="center">
    <img src='./gifs/DCGAN.gif' width="360"\>
</p>