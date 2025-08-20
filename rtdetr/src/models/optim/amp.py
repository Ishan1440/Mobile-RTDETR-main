import torch
import torch.nn as nn
import torch.cuda.amp as amp


__all__ = ['GradScaler']

GradScaler = amp.grad_scaler.GradScaler
