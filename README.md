# _*Pytorch Implementations*_

[Code Reference](https://github.com/eriklindernoren/PyTorch-GAN)   I've referred to this page a lot.


Implementing Serveral Networks in [**pytorch**](https://pytorch.org) with studying it each papers. Welcome any advice with widely open arms.

<br></br>

[**Implementations**](#Implementations)
 + [GANs](#Gans)
   + [VanilaGAN](#vanilagan)
   + [DCGAN](#dcgan)
   + [LSGAN](#lsgan)

<br></br>

#### VanilaGAN
- Generative Adversarial Network
- Authors 
  - [Ian J. Goodfellow | Jean Pouget-Abadie | Mehdi Mirza | Bing Xu | David Warde-Farley | Sherjil Ozair | Aaron Courville | Yoshua Bengio]
  <br></br>
- [[**Paper**]](https://arxiv.org/abs/1406.2661) | [[**Code**]](./Implementations/GANs/VanilaGAN/VanilaGAN.ipynb)
<p align="center">
    <img src='./gifs/VanilaGAN.gif' width="360"\>
</p>

#### DCGAN
- Deep Convolutional Generative Adversarial Networks
- Authors 
  - [Alec Radford | Luke Metz | Soumith Chintala]
  <br></br>
<p align="center">
    <img src="./imgs/DCGAN.png" width="400" height="250"\>
</p>

- The Main Idea is embedding CNN into the VanilaGAN
- [[**Paper**]](https://arxiv.org/abs/1511.06434) | [[**Code**]](./Implementations/GANs/DCGAN/DCGAN.ipynb)
<p align="center">
    <img src='./gifs/DCGAN.gif' width="360"\>
</p>

#### LSGAN
- Least Squares Generative Adversarial Networks
- Authors 
  - [Xudong Mao | Qing Li | Haoran Xie | Raymond Y.K. Lau | Zhen Wang | Stephen Paul Smolley]
  <br></br>
<p align="center">
    <p align="center">
        <font size="3.5">
          The VanilaGANs Loss Function
        </font>
    </p>
    <p align="center">
    <img src="./imgs/LSGAN_2.png" width="400"\>
    </p>
    <p align="center">
        <font size="3.5">
        The LSGANs Loss Function
        </font>
    </p>
    <p align="center">
    <img src="./imgs/LSGAN_1.png" width="400"\>
    </p>
</p>

- The authors claim that VanilaGAN is UNSTABLE cause of the loss function. Breifly, minimizing the objective function of it suffers from vanishing gradients and it ends up with being hard to train the generator. To Resolve this problem, the authors argue <font color="red" size="3.5"> **"The least squares loss function will penalize the fake samples and pull them toward the decision boundary even though they are correctly classfied. Based on this porperty, LSGANs are able to generate samples that are closer to real data."** </font>
- [[**Paper**]](https://arxiv.org/abs/1611.04076) | [[**Code**]](./Implementations/GANs/LSGAN/LSGAN.ipynb)
<p align="center">
    <img src='./gifs/LSGAN.gif' width="360"\>
</p>