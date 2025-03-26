#!/usr/bin/env python3

import os
import argparse

from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet
from huggingface_hub import login
from huggingface_hub import HfApi

if __name__ == '__main__':
    models = {
        'lenet': LeNet,
        'pure_cnn': PureCnn,
        'net_in_net': NetworkInNetwork,
        'resnet': ResNet,
        'densenet': DenseNet,
        'wide_resnet': WideResNet,
    }

    parser = argparse.ArgumentParser(description='Train models on Cifar10')
    parser.add_argument('--model', choices=models.keys(), default=None, help='Specify a model by name to train.')
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--bulk', default=False, action='store_true', help='Use bulk training.')
    parser.add_argument('--upload', default=False, action='store_true', help='Upload models to cloud.')

    args = parser.parse_args()
    model_name = args.model
    bulk = args.bulk
    upload = args.upload
    args = {k: v for k, v in vars(args).items() if v != None}

    # Delete the model key from args if it exists
    if 'model' in args:
        del args['model']
    if 'bulk' in args:
        del args['bulk']
    if 'upload' in args:
        del args['upload']

    if bulk:
        for k, v in models.items():
            model = v(**args, load_weights=False)
            model.train()
    elif model_name:
        model = models[model_name](**args, load_weights=False)
        model.train()

    if upload:
        login(
            new_session=False, # Won’t request token if one is already saved on machine
            write_permission=True, # Requires a token with write permission
            token=os.environ['HF_TOKEN_WRITE']
        )
        
        api = HfApi()
        api.upload_folder(
            folder_path='./networks/models', 
            repo_id="Ethics2025W/base",
        )

