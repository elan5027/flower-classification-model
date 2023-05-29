import os, json
with open('flower_to_name.json', 'r') as f:
	flower_to_name = json.load(f)
path = 'rename/train/'

for i in os.listdir(path):
    flower = flower_to_name[i].replace(' ','_')
    os.rename(path+i, path+flower)
    


