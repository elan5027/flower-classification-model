import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
import torch
import json
from PIL import Image
#pytorch, numpy, matplotlib, PIL


def image_preprocessing(img_pth):
	
	pil_image = Image.open(img_pth)
	
	if pil_image.size[0] > pil_image.size[1]:
		pil_image.thumbnail((10000000, 256))
	else:
		pil_image.thumbnail((256, 100000000))
	
	left_margin = (pil_image.width - 224) / 2
	bottom_margin = (pil_image.height - 224) / 2
	right_margin = left_margin + 224
	top_margin = bottom_margin + 224
	
	pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
	
	np_image = np.array(pil_image) / 255
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	np_image = (np_image -mean) / std
	
	np_image = np_image.transpose([2, 0, 1])
	
	return np_image


def load_model(chkpt_path):
	
	chkpt = torch.load(chkpt_path)
	model = models.vgg19(pretrained = True)
	
	for params in model.parameters():
		params.requires_grad = False
	
	model.classifier = chkpt['classifier']
	model.load_state_dict(chkpt['state_dict'])
	
	return model

def predict(image_path, model, idx_class_mapping, device, topk=5):
    model = load_model(model)
    model.to(device)
    
    model.eval()
     
    img = image_preprocessing(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    with torch.no_grad():
        log_probabilities = model.forward(img_tensor)
    
    probabilities = torch.exp(log_probabilities)
    probs, indices = probabilities.topk(topk)
    print(indices)
    if torch.cuda.is_available():
        probs = probs.cpu()
        indices = indices.cpu()
    probs = probs.numpy().squeeze()
    indices = indices.numpy().squeeze()
    classes = [idx_class_mapping[index] for index in indices]
    
    return probs, classes

deviceFlag = torch.device('cpu')

if torch.cuda.is_available():
	print(f'Found {torch.cuda.device_count()} GPUs.')
	deviceFlag = torch.device('cuda:0')

data_dir = 'flowers'
train_dir = data_dir + '/train'

training_transforms = transforms.Compose([
	transforms.RandomRotation(30),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(), 
	transforms.Normalize([0.485, 0.456, 0.406], # RGB mean & std estied on ImageNet
						 [0.229, 0.224, 0.225])
])


# # Load the datasets with torchvision.datasets.ImageFolder object
training_imagefolder = datasets.ImageFolder(train_dir, transform = training_transforms)

with open('flower_to_name.json', 'r') as f:
	flower_to_name = json.load(f)

class_idx_mapping = training_imagefolder.class_to_idx
idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}
img_path = "01.jpg"
# probs, classes = predict(image_path=img_path, model="chkpt.pth", device=deviceFlag, idx_class_mapping=idx_class_mapping)
probs, classes = predict(image_path=img_path, model="chkpt.pth", device=deviceFlag, idx_class_mapping=idx_class_mapping)
class_names = [flower_to_name[c] for c in classes]

flower_match = []
for name in zip(class_names, probs):
    flower_match.append(name)
    
print(flower_match)
