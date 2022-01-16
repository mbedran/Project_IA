# -*- coding: utf-8 -*-
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import network
import datasets_ws

#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = join("runs", args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset StLucia from folder {args.datasets_folder}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, "st_lucia", "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = network.GeoLocalizationNet(args)
model = model.to(args.device)

#### Setup Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

best_r5 = 0
not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.output_folder, "/content/drive/MyDrive/IA_project/project_vg/runs/default/GeM_pooling/best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")