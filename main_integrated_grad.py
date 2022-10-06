import os
import cv2

from integrated_grad import IntegratedGradients

from parameters import Params
from paths import Path
from data import ClsDataLoader, GraspDataLoader
from utils import tensor2img
from grasp_utils import visualize_grasp
from misc_functions import convert_to_grayscale, save_gradient_images

# Params class containing parameters for AM visualization.
params = Params()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = Path('gr-convnet')
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
    # alexnetCLS
    ext_model = model
    # Integrated backprop
    IG = IntegratedGradients(ext_model)
    prep_img = img
    # Generate gradients
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
