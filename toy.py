import argparse
import os
import sys
import json
import shutil
import pathlib
import random
from itertools import islice
import time
from copy import deepcopy
import math
from pprint import pprint
from qj_global import qj
import logging

try:
    from comet_ml import Experiment
    COMET_AVAIL = True
except:
    COMET_AVAIL = False

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, Timer, TerminateOnNan, TimeLimit
from ignite.metrics import RunningAverage
from ignite.contrib.metrics import GpuInfo
from ignite.utils import setup_logger

# adding the folder containing the folder `disentanglement_via_mechanism_sparsity` to sys.path
# sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str('/Users/vitoria/Documents/GitHub/'))
from disentanglement_via_mechanism_sparsity.universal_logger.logger import UniversalLogger
from disentanglement_via_mechanism_sparsity.metrics import MyMetrics, linear_regression_metric, mean_corr_coef, edge_errors
from disentanglement_via_mechanism_sparsity.plot import plot_01_matrix
from disentanglement_via_mechanism_sparsity.data.synthetic import get_ToyManifoldDatasets
from disentanglement_via_mechanism_sparsity.model.ilcm_vae import ILCM_VAE
from disentanglement_via_mechanism_sparsity.model.latent_models_vae import FCGaussianLatentModel


manifold = "nn"
transition_model = "action_sparsity_trivial"

datasets = get_ToyManifoldDatasets(manifold, transition_model, split=(0.8, 0.1, 0.1),
                                       z_dim=10, x_dim=20, num_samples=int(1e6),
                                       no_norm=True, discrete=True)