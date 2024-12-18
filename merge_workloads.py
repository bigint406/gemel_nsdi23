import os
import sys
import json
import torch
import time
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.hub import load_state_dict_from_url
import torch.nn as nn
from model_merger import ModelMerger
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from collections import OrderedDict
from itertools import combinations

from transforms import classification_transforms
from eval_methods import frcnn_eval, classification_eval
from models.model_architectures import resnet50_backbone, resnet101_backbone, frcnn_model, resnet50, resnet101, vgg16, resnet152, resnet18, mobilenetv3, inceptionv3, ssd_vgg_model, ssd_mobilenet_model, yolov3, tiny_yolov3

# Expects a dict containing each model to be merged and some info about it (see README and example below)
def merge_workload(model_dict):
	for key in model_dict.keys():
		unmerged_acc = model_dict[key]['unmerged_acc']
		print(f'{key}: {unmerged_acc}')
	
	# Store the merged results in a results folder
	path = f'results'
	if not os.path.exists(path):
		os.mkdir(path)

	merger = ModelMerger(model_dict, path)
	merger.merge()

# An example of a model dict with two tasks (entries)
# One classifies cars vs. people at Main & 2nd with ResNet50
# One classifies cars, trucks, and motorcycles at 1st & Elm with ResNet101
def create_sample_model_dict():
	tasknames = ['main_2nd_cat_fish_resnet50', 'elm_1st_car_truck_train_resnet101']
	model_dict = {}
	
	# Task 1
	model_dict[tasknames[0]] = {}
	model_dict[tasknames[0]]['unmerged_acc'] = 0.97

	# Initialize model structure and load weights
	model_main_2nd = resnet50(2) # 2 classes
	# model_main_2nd.load_state_dict(torch.load('tasknames[0]_weights.pt'))
	model_dict[tasknames[0]]['model'] = model_main_2nd

	model_dict[tasknames[0]]['task'] = 'image_classification'
	model_dict[tasknames[0]]['eval_method'] = classification_eval
	model_dict[tasknames[0]]['transforms'] = {'train': classification_transforms, 'val': classification_transforms}

	# Task 2
	model_dict[tasknames[1]] = {}
	model_dict[tasknames[1]]['unmerged_acc'] = 0.99

	# Initialize model structure and load weights
	model_elm_1st = resnet101(3) # 3 classes
	# model_elm_1st.load_state_dict(torch.load('tasknames[1]_weights.pt'))
	model_dict[tasknames[1]]['model'] = model_elm_1st

	model_dict[tasknames[1]]['task'] = 'image_classification'
	model_dict[tasknames[1]]['eval_method'] = classification_eval
	model_dict[tasknames[1]]['transforms'] = {'train': classification_transforms, 'val': classification_transforms}

	return model_dict

def main():
	model_dict = create_sample_model_dict()
	merge_workload(model_dict)

if __name__ == "__main__":
	main()