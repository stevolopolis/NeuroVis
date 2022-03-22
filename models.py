import torch.nn as nn

class ExtractModel(nn.Module):
    def __init__(self, model, layer, net_type='gr-convnet', device='cuda'):
        super().__init__()
        self.output_layer_name = layer
        self.children_list = []

        # for VGG
        if net_type == 'vgg16':
            for n, c in model.features.named_children():
                self.children_list.append(c)
                if n == layer:
                    break
        # for gr-convnet
        elif net_type == 'gr-convnet':
            for n,c in model.named_children():
                self.children_list.append(c)
                #if n[:2] == 'bn':
                #    self.children_list.append(nn.ReLU())
                if n == layer:
                    break
        # for resnet18
        elif net_type == 'resnet18':
            for n, c in model.named_children():
                self.children_list.append(c)
                if n == layer:
                    break
        
        self.net = nn.Sequential(*self.children_list).to(device)
        
    def forward(self,x):
        x = self.net(x)
        return x