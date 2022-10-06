"""
This code visualizes the saliency map of a CNN using integrated gradients,
which highlights which areas in an image influences the model prediction
the most. 

For classification models, we take the class label as the ground
truth for calculating gradients. For grasping models, we take the 
'cos' output of the final layer as the ground truth. Other outputs
such as the 'width', 'sin', etc. could also be used if required.

The saliency maps are generated from the Jacquard Dataset.
The images are saved in the <datasets> folder.

This file has referenced codes from @author: Utku Ozbulak - github.com/utkuozbulak.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""
from hypothesis import target
import torch
import torch.nn as nn

from parameters import Params
from misc_functions import (convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

# Params class containing parameters for AM visualization.
params = Params()

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_fn(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        # first_layer = list(self.model._modules.items())[0][1]
        #print(self.model.net)
        # For Alexnet
        first_layer = self.model
        # For gr-convnet
        # first_layer = self.model.net
        first_layer.register_full_backward_hook(hook_fn)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, kernel_idx):
        self.model.zero_grad()
        # Forward pass
        input_image.requires_grad_()
        x = self.model(input_image)
        x = x[0, kernel_idx]
        conv_output = torch.sum(torch.abs(x))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    import os
    import cv2

    from paths import GrPath
    from data_v2 import DataLoaderV2
    from models import AlexnetMapRgbdFeatures, AlexnetMapFeatures
    from utils import tensor2img, get_layer_width

    # Path class for managing required directories
    # gr-convnet / resnet18 / vgg16
    paths = GrPath('gr-convnet')
    paths.create_guided_am_path()

    # Select data for visualization
    data = []
    # DataLoader
    dataLoader = DataLoaderV2('datasets/top_5_compressed_paperspace/test', 1)
    for i, datum in enumerate(dataLoader.load_grasp()):
        if i >= params.N_IMG:
            break
        data.append(datum)

    # Trained model paths
    model = params.MODEL
    for vis_layer in params.vis_layers:
        print('Visualizing for %s layer' % vis_layer)
        # Create sub-directory for chosen layer
        paths.create_layer_paths(vis_layer)
        
        # Create submodel with output = selected kernel
        #ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
        ext_model = AlexnetMapFeatures(model, vis_layer)
        n_kernels = get_layer_width(ext_model)

        # AM on individual kernels
        for kernel_idx in range(n_kernels):
            paths.create_kernel_paths(vis_layer, str(kernel_idx))

            # Load Jacquard Dataset images
            print('Visualizing saliency maps')
            for img, map, img_id in data:
                # Create img folder
                #paths.create_img_paths(img_id)
                # Image preprocessing (None for now)
                prep_img = img
                
                GBP = GuidedBackprop(ext_model)
                # Get gradients
                guided_grads = GBP.generate_gradients(prep_img, kernel_idx)
                guided_grads = guided_grads[:3]
                # Convert to grayscale
                grayscale_guided_grads = convert_to_grayscale(guided_grads)
                # Save grayscale gradients
                save_gradient_images(grayscale_guided_grads, paths.save_subdir,
                                     '%s_%s_%s' % (img_id, vis_layer, kernel_idx) + '_Guided_BP_gray' )
                save_img_rgb, save_img_d = tensor2img(img)
                if '%s_image.png' % img_id not in os.listdir(os.path.join(paths.vis_path, paths.main_path)):
                    cv2.imwrite(os.path.join(paths.vis_path, paths.main_path, '%s_image.png' % img_id), save_img_rgb)
        