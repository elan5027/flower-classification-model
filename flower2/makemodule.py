

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

from collections import OrderedDict

# ---------------------- HYPERPARAMS ------------------------ #
tr_batchsize = 32
epochs = 10 #15로 햇을때 결과가 좋았음.
lr = 0.0001

# 기본적으로 CPU를 사용하도록 디바이스 설정
deviceFlag = torch.device('cpu')

if torch.cuda.is_available():
	print(f'발견한 GPU 수: {torch.cuda.device_count()}개')
	deviceFlag = torch.device('cuda:0')

print(f'현재 디바이스 설정: {deviceFlag}')


def validation(model, validateloader, ValCriterion):
	# 모델에 대한 검증(Validation) 수행을 위한 함수
	val_loss_running = 0
	acc = 0
	model.to(deviceFlag)
	model.eval()
 
	with torch.no_grad():
	# 데이터로더 객체는 이미지와 레이블을 개별적으로 포함하는 배치의 생성기.
		for images, labels in validateloader:
			
			# 데이터를 선택한 디바이스로 보낸다.
			images = images.to(deviceFlag)
			labels = labels.to(deviceFlag)
			
			output = model.forward(images)
			val_loss_running += ValCriterion(output, labels).item() # Torch.tensor에서 스칼라를 얻기 위해 .item()을 사용함.
			
			output = torch.exp(output) 
			
			equals = (labels.data == output.max(dim = 1)[1])
			acc += equals.type(torch.FloatTensor).mean()
		
	return val_loss_running, acc 

def train_eval(model, traindataloader, validateloader, criterion, optimizer, epochs, deviceFlag_train, print_every):
    steps = 0
    model.train()
    model.to(deviceFlag_train)
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in traindataloader:
            steps += 1
            images = images.to(deviceFlag_train)
            labels = labels.to(deviceFlag_train)
            # 이전 반복에서 계산된 grads를 지운다.
            optimizer.zero_grad()
            outputs = model.forward(images)
            train_loss = criterion(outputs, labels)
            # grads를 계산하기 위해 BackProp 수행 (각 tensor.grad() 속성에 저장됨)

            train_loss.backward()
            
            # Optimizer / 파라미터 업데이트
            optimizer.step()
            
			# Forward Pass
            # 모델이 대상 디바이스로 이동했으므로 출력도 해당 디바이스에 존재.
            running_loss += train_loss.item() # numeric ops, .item()을 호출하여 tensor에서 스칼라를 얻는다.
			
			# ----------- 일정 주기로 검증(Evaluation) 수행 ---------- #
            if steps % print_every == 0:
                
				# 모델을 Eval 모드로 설정한다.
                model.eval()
				
				# 검증을 위해 gradient를 끈다. (메모리와 계산 시간을 절약하기 위해)
                with torch.no_grad():
                    validation_loss, val_acc = validation(model, validateloader, criterion)
                
                print("{}/{} 배치: {}/{}.. ".format(steps % len(traindataloader), len(traindataloader), e + 1, epochs),
                      "훈련 손실: {:.3f}.. ".format(running_loss / print_every),
                      "검증 손실: {:.3f}.. ".format(validation_loss / len(validateloader)),
                      "검증 정확도: {:.3f}".format((val_acc / len(validateloader)) * 100))
                running_loss = 0
                model.train()
    		

# ------------------------ DATA LOADING ----------------------- #
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


# ----------------------- DATA AUGMENTATION ---------------------- #
training_transforms = transforms.Compose([
	transforms.RandomRotation(30),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(), 
	transforms.Normalize([0.485, 0.456, 0.406], 
						 [0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], 
						 [0.229, 0.224, 0.225])
])

# torchvision.datasets.ImageFolder 객체를 사용하여 데이터셋 로드
training_imagefolder = datasets.ImageFolder(train_dir, transform = training_transforms)
validation_imagefolder = datasets.ImageFolder(valid_dir, transform = validation_transforms)

train_loader = torch.utils.data.DataLoader(training_imagefolder, batch_size = tr_batchsize, shuffle = True)
validate_loader = torch.utils.data.DataLoader(validation_imagefolder, shuffle = True, batch_size = tr_batchsize)

# ------------------------- CLASS LABELLING -------------------------- #
import json
with open('flower_to_name.json', 'r') as f:
	flower_to_name = json.load(f)


# # -------------------------- 사전 훈련된 모델에 새로운 헤드 추가하여 사전 훈련 ------------------------ #
model = models.vgg19(pretrained = True)

for params in model.parameters():
	params.requries_grad = False


NewClassifier = nn.Sequential(OrderedDict([
	('fc1', nn.Linear(25088, 4096)),
	('relu', nn.ReLU()),
	('drop', nn.Dropout(p = 0.5)),
	('fc2', nn.Linear(4096, 102)),
	('output', nn.LogSoftmax(dim = 1))
]))

model.classifier = NewClassifier
model.class_idx_mapping = training_imagefolder.class_to_idx


# # ------------------------- LOSS & OPTIMIZER DEF ----------------------------- #
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = lr)


# # ----------------------------- TRAINING ----------------------------------- #
print('**************** Training begins *****************')
train_eval(
    model=model, 
    traindataloader=train_loader, 
    validateloader=validate_loader, 
    criterion=criterion, 
    optimizer=optimizer, 
    epochs=epochs, 
    deviceFlag_train=deviceFlag, 
    print_every=20)


# # ----------------------------- Chkpt SAVING ----------------------------------- #
print('**************** Saving Checkpoint ****************')
state = {
	'epoch':epochs,
	'classifier': model.classifier,
	'state_dict': model.state_dict(),
	'optimizer'	: optimizer.state_dict(),
	'class_idx_mapping' : training_imagefolder.class_to_idx,
	'arcg' : "vgg19"
}
torch.save(state, 'chkpt.pth')

