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
import torch.nn.functional as F
import numpy as np

from inference.models.grconvnet_cls import GrCLS
from parameters import GrParams
from misc_functions import convert_to_grayscale, save_gradient_images

# Params class containing parameters for AM visualization.
params = GrParams()

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


if __name__ == '__main__':
    import os
    import cv2

    from paths import GrPath
    from data import DataLoader, jacquard_sin_loader, ClsDataLoader, GraspDataLoader
    from models import ExtractOutputModel, ExtractModel, ExtractAlexModel
    from inference.models.alexnet import AlexNet
    from utils import tensor2img
    from grasp_utils import visualize_grasp

    # Path class for managing required directories
    # gr-convnet / resnet18 / vgg16
    paths = GrPath('gr-convnet')
    paths.create_grad_path()

    # Trained model paths
    model = params.MODEL

    # DataLoader for Jacquard Dataset
    #dataLoader = ClsDataLoader(randomize=True)
    dataLoader = GraspDataLoader(randomize=True)
    # Load Jacquard Dataset images
    print('Visualizing saliency maps')
    #for i, (img, img_id, img_cls) in enumerate(dataLoader.load()):
    for i, (img, label, label_list, img_id) in enumerate(dataLoader.load_grasp()):
        if i > 25:
            break
        paths.create_img_paths(img_id)
        # Model with only 'cos' output
        # gr-convnet
        #ext_model = ExtractOutputModel(model, 'sin')
        # gr-convnet alternative
        #ext_model = ExtractModel(model, 'sin', net_type='gr-convnet-sin')
        # alexnetCLS
        ext_model = model
        # Integrated backprop
        IG = IntegratedGradients(ext_model)
        # Normalize image with training mean and std
        #prep_img = imagenet_norm(img)
        prep_img = img
        # Generate gradients
        #target_class = jacquard_sin_loader(img_id)
        integrated_grads = IG.generate_integrated_gradients(prep_img, label, False, 100)
        # Take only RGB out of RGBD channels -- (4, OUTPUT_SIZE, OUTPUT_SIZE) --> (3, OUTPUT_SIZE, OUTPUT_SIZE)
        integrated_grads = integrated_grads[:3]
        # Convert to grayscale
        grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
        # Save grayscale gradients
        save_gradient_images(grayscale_integrated_grads, os.path.join(params.GRAD_PATH, img_id),
                             img_id + '_Integrated_G_gray')
        #save_img_rgb, save_img_d = tensor2img(prep_img)
        save_img_rgb = visualize_grasp(model, prep_img, label)
        cv2.imwrite(os.path.join('vis', params.MODEL_NAME, params.GRAD_PATH, img_id, 'image.png'), save_img_rgb)
        print('Integrated gradients completed.')
