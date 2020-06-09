from torchvision import transforms
import torch
from PIL import Image
import numpy as np 
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
def to_tensor(img):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    image_tensor = Variable(transform(img)).to(device).unsqueeze(0)
    return image_tensor
