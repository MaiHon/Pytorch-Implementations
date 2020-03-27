import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from visdom import Visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from IPython.display import clear_output


viz = Visdom()
viz.close(env='main')

def loss_tracker(loss_plot, loss_value, num):
    viz.line(X=num, Y=loss_value, win=loss_plot, update='append')
    

class G(nn.Module):
    def __init__(self, noise=128):
        super(G, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.G = torch.nn.Sequential(
            torch.nn.Linear(noise, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 28*28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.G(x)

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.D = torch.nn.Sequential(
            torch.nn.Linear(28*28, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.D(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Gen = G().to(device)
Dis = D().to(device)

from torchsummary import summary
summary(Gen, (128, ), batch_size=100, device=device)
summary(Dis, (28*28, ), batch_size=100, device=device)


g_optimizer = torch.optim.Adam(Gen.parameters(), 2e-4)
d_optimizer = torch.optim.Adam(Dis.parameters(), 2e-4)


# Train Dataset Prepare
dataset = dsets.MNIST('./MNIST', 
                    train=True, 
                    transform=transforms.ToTensor(),
                    target_transform=None,
                    download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)

# Fixed Noise for Testing

test_noise = Variable(torch.randn(10, 128)).to(device)
image_window = viz.images(torch.randn(10, 1, 28, 28), 
                        opts=dict(title = "Generated Imgs",
                        caption = "Generated Image-{}-{}".format(0, 0)))
loss_plt = viz.line(Y=torch.randn(1, 2).zero_(), 
                    opts=dict(title='Tracking Losses',
                    legend=['D_Loss', 'G_Loss'], 
                    showlegend=True)
)


total_step = 0
total_batch = len(data_loader)

for epoch in range(200):
    for step, data in enumerate(data_loader):
        images = data[0]
        images = images.to(device)

        # Train D
        noise = Variable(torch.randn(images.size(0), 128))
        noise = noise.to(device)
        fake_images = Gen(noise)
        dis_fake_results = Dis(fake_images)
        dis_real_results = Dis(images.reshape(-1, np.prod(images.shape[1:])))

        d_loss = -torch.mean(torch.log(dis_real_results) + torch.log(1-dis_fake_results))
        Dis.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        # Train G
        noise = Variable(torch.randn(images.size(0), 128))
        noise = noise.to(device)
        fake_images = Gen(noise)
        dis_fake_results = Dis(fake_images)
        g_loss = - torch.mean(torch.log(dis_fake_results) + 1e-6)

        Gen.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        total_step += 1
        
        # Print & Showing via Visdom
        if (step + 1) % 10 == 0:
            clear_output(wait=True)
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'% (epoch + 1, 200, step + 1, total_batch, d_loss.data, g_loss.data))
            fake_images = Gen(test_noise)
            fake_images = fake_images.reshape(10, 1, 28, 28)
            loss_tracker(loss_plt, np.column_stack((d_loss.detach().cpu().data, g_loss.detach().cpu().data)), 
                        np.column_stack((torch.Tensor([total_step]), 
                        torch.Tensor([total_step]))))
            image_window = viz.images(fake_images.data,
                                    opts=dict(title = "Generated Imgs",
                                    caption = "Generated Image-{}-{}".format(epoch + 1, step + 1)),
                                    win = image_window
            )
        
        # Image Save
        if (epoch + 1) % 10 == 0 and (step+1) == total_batch:
            fake_images = Gen(test_noise)
            fake_images = fake_images.reshape(10, 1, 28, 28).detach()
            save_image(fake_images.data, './IMGS/generatedimage-%d-%d.png' % (epoch + 1, step + 1))