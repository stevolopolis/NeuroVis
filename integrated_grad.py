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
import torch
import torch.nn.functional as F
import numpy as np

from parameters import Params

from parameters import Params

# Params class containing parameters for AM visualization.
params = Params()
# Trained model paths
model = params.MODEL

class IntegratedGradients():
    """
        Produces gradients generated with integrated gradients from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
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

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and steps
        step_list = np.arange(steps+1)/steps
        # Generate scaled xbar images
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, target_class, cls):
        # Zero grads
        self.model.zero_grad()
        # Forward --> (1, 1, OUTPUT_SIZE, OUTPUT_SIZE)
        input_image.requires_grad_()
        model_output = self.model(input_image)
        model_output = F.softmax(model_output, dim=1)
        # Target for backprop (cls model)
        if cls:
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1], device=params.DEVICE).zero_()
            one_hot_output[0][target_class] = 1
        # Target for backprop (grasp model -- 'cos' map)
        else:
            one_hot_output = target_class.to(params.DEVICE)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, cls, steps):
        # Generate xbar images
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        # Initialize an iamge composed of zeros
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            # Generate gradients from xbar images
            single_integrated_grad = self.generate_gradients(xbar_image, target_class, cls)
            # Add rescaled grads from xbar images
            integrated_grads = integrated_grads + single_integrated_grad/steps
        # [0] to get rid of the first channel (1,3,224,224)
        return integrated_grads[0]
