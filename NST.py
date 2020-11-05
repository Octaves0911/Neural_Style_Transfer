# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 02:21:36 2020

@author: amanm
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image



model=models.vgg19(pretrained=True).features


device=torch.device("cuda")
#[0,5,10,19,28]

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','5','10','19','28']
        self.model=models.vgg19(pretrained=True).features[:29]
        
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            
            if (str(layer_num) in self.req_features):
                features.append(x)
                
        return features

loader=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])

def image_loader(path):
    image=Image.open(path)
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)


original_image=image_loader('content image.jpg')
style_image=image_loader('style image.jpg')

generated_image=original_image.clone().requires_grad_(True)
model=VGG().to(device).eval()

epoch=1000
lr=0.001
alpha=1
beta=0.01

optimizer=optim.Adam([generated_image],lr=lr)




for e in range (epoch):
    gen_features=model(generated_image)
    orig_feautes=model(original_image)
    style_featues=model(style_image)
    
    content_loss=style_loss=0
    
    for gen,cont,style in zip(gen_features,orig_feautes,style_featues):
        batch_size,channel,height,width=gen.shape
        content_loss+=torch.mean((gen-cont)**2)
        
        G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
        A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
        
        style_loss+=torch.mean((G-A)**2)
   
    total_loss=alpha*content_loss + beta*style_loss    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if(e%100):
        print(total_loss)
        save_image(generated_image,"gen.png")
    
        
        
    
        

