import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
from models.archs.arch_util import feature_fusion
import numpy as np
import cv2

from models.archs.arch_util import Down
from models.archs.transformer.Models import Encoder_patch66

###############################
