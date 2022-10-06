"""
This file contains the <ActivationMaximization> class which
implements AM on selected kernels for a pretrained model.

Regularizer losses are also available for improved AM
results. The regularizers include:
    - LP loss
    - Jitter loss
    - Total variation loss

The class also creates a 2x3 image matrix that contains:
    - initial image (RGB + Depth)
    - backpropagated image (RGB + Depth)
    - target feature map (i.e. fully activated pixels)
    - feature map of backpropagated image
"""

import torch 
import cv2
from torchvision import transforms
from utils import tensor2img

from am_regularizers import LPLoss, TVLoss
from parameters import GrParams

params = GrParams()


class ActivationMaximization():
    """This class performs AM on a pretrained model.
    
    Users can specify the img_size, lr, and number of 
    iterations (epochs).
    
    Users can also specify the init_method:
        - noise -- initialize a guassian noise image for backprop
        - zero -- initialize a black image for backprop

    For torchvision models, a preprocessing pipeline is 
    also included.
    """
    def __init__(self, model, img_size, lr=1e-3, epochs=100,
                 init_method='noise', device='cuda'):
        self.model = model
        self.img_size = (1, img_size[0], img_size[1], img_size[2])
        self.device = device

        self.lr = lr
        self.epochs = epochs
        self.loss = torch.tensor(0.0)

        # Preprocessing pipeline for torch.models
        self.preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def am(self, kernel_idx):
        pixel_start, pixel_backprop, pixel_fmap, pixel_target = self.backprop_pixel(kernel_idx)
        full_start, full_backprop, full_fmap, full_target = self.backprop_full(kernel_idx)
        pixel_result_set = self.process_results(pixel_start, pixel_backprop, pixel_fmap, pixel_target)
        full_result_set = self.process_results(full_start, full_backprop, full_fmap, full_target)
       
        return pixel_result_set, full_result_set

    def backprop_full(self, kernel_idx, alpha=.1, beta=.1, p=2):
        """Implements AM on <self.backprop_img> on a selected kernel.

        Parameters:
            - kernel_idx: index of visualizing kernel
            - alpha: weight for pixel-value loss
            - beta: weight for total-variation loss
            - sigma: weight for jitter loss
            - p: raising power for LP loss
        """
        if params.INIT_METHOD == 'noise':
            backprop =  torch.randn(self.img_size, dtype=torch.float32,
                                    requires_grad=True, device=self.device)
        elif params.INIT_METHOD == 'zero':
            backprop =  torch.full(self.img_size, 0.0, dtype=torch.float32,
                                    requires_grad=True, device=self.device)
        start = backprop.detach().clone()
        optim = torch.optim.SGD([backprop], lr=self.lr, momentum=0.99)

        for _ in range(self.epochs):
            optim.zero_grad()

            # for vgg / resnet
            if params.net in ('vgg16', 'resnet18'):
                input = self.preprocess(backprop.squeeze()).unsqueeze(0)
            # for gr-convnet
            elif params.net == 'gr-convnet':
                input = backprop
            
            output = self.model(input)
            fmap_output = output[0, kernel_idx]

            pixel_loss = -torch.sum(fmap_output) \
                            + TVLoss(backprop) * beta \
                            + LPLoss(fmap_output, p=p) * alpha
            loss = pixel_loss

            loss.backward()
            optim.step()

        # Target feature map with maxed out pixel values
        target = torch.ones(fmap_output.shape)
        target[0][0] = 0

        # Display AM loss
        self.show_loss(loss)

        return start, backprop, fmap_output, target

    def backprop_pixel(self, kernel_idx):
        """Implements pixel-AM on <self.backprop_img> on a selected kernel.

        Pixel-AM is done by calculating losses on 1 center pixel.
        """
        if params.INIT_METHOD == 'noise':
            backprop =  torch.randn(self.img_size, dtype=torch.float32,
                                    requires_grad=True, device=self.device)
        elif params.INIT_METHOD == 'zero':
            backprop =  torch.full(self.img_size, 0.0, dtype=torch.float32,
                                    requires_grad=True, device=self.device)
        start = backprop.detach().clone()
        optim = torch.optim.SGD([backprop], lr=self.lr, momentum=0.99)

        for _ in range(self.epochs):
            optim.zero_grad()

            # for vgg / resnet
            if params.net in ('vgg16', 'resnet18'):
                input = self.preprocess(backprop.squeeze()).unsqueeze(0)
            # for gr-convnet
            elif params.net == 'gr-convnet':
                input = backprop

            output = self.model(input)
            fmap_output = output[0, kernel_idx]
            
            # Getting center coordinate of feature map
            h = fmap_output.shape[0]
            w = fmap_output.shape[1]
            x_mid = w // 2
            y_mid = h // 2

            loss = - fmap_output[y_mid, x_mid]

            loss.backward()
            optim.step()

        # Target feature map with maxed out pixel values
        target = torch.zeros(fmap_output.shape)
        target[y_mid, x_mid] = 1.0

        # Display AM loss
        self.show_loss(loss)

        return start, backprop, fmap_output, target

    def backprop_pred(self, idx):
        if params.INIT_METHOD == 'noise':
                backprop =  torch.randn(self.img_size, dtype=torch.float32,
                                        requires_grad=True, device=self.device)
        elif params.INIT_METHOD == 'zero':
            backprop =  torch.full(self.img_size, 0.0, dtype=torch.float32,
                                    requires_grad=True, device=self.device)
        start = backprop.detach().clone()
        optim = torch.optim.SGD([backprop], lr=self.lr, momentum=0.99)

        for _ in range(self.epochs):
            optim.zero_grad()
            output = self.model(backprop)
            loss = - output[0][idx]
            
            loss.backward()
            optim.step()

        # Display AM loss
        self.show_loss(loss)

        return start, backprop

    def show_loss(self, loss):
        """Print loss value for current backprop step."""
        rounded_loss = round(loss.item(), 5)
        print('Epoch: %s\t Loss: %s' % (self.epochs, rounded_loss))

    def process_results(self, start, backprop, fmap_output, target):
        """Return a tuple of 6 images after AM is run.

        Return images:
            - start_img_rgb: RGB image of initial noisy/zero input image
            - start_img_d: Depth image of initial noisy/zero input image
            - backprop_img_rgb: RGB image of input image after AM backprop
            - backprop_img_d: Depth image of input image after AM backprop
            - target: image of target feature map for selected kernel
            - output: image of feature map of selected kernel
        """
        # Convert tensors to images compatible for cv2.imshow/cv2.imwrite
        start_rgb, start_d = tensor2img(start, cv2.INTER_LINEAR)
        backprop_rgb, backprop_d = tensor2img(backprop, cv2.INTER_LINEAR)
        target, _ = tensor2img(target)
        fmap, _ = tensor2img(fmap_output)
        
        return start_rgb, start_d, \
                backprop_rgb, backprop_d, \
                target, fmap