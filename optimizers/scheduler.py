import math

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class CosineAnnealingScheduler(LambdaLR):
    def __init__(self, optimizer, total_steps, lr_base, lr_min, lr_max=1):
        """
        Cosine Annealing Scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of training steps.
            lr_min (float): Minimum learning rate at the end of annealing.
            lr_max (float): Maximum learning rate at the start of annealing.
        """
        self.total_steps = total_steps
        self.lr_base = lr_base
        self.lr_min = lr_min / self.lr_base
        self.lr_max = lr_max

        # Define the lambda function for learning rate adjustment
        lr_lambda = lambda step: self.cosine_annealing(step)
        super().__init__(optimizer, lr_lambda=lr_lambda)

    def cosine_annealing(self, step):
        """Cosine Annealing for learning rate decay scheduler"""
        return self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
                1 + np.cos(step / self.total_steps * np.pi))


class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0,
                 warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]

        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (
                    self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

        self._last_lr = [value]
        return value

    def step_func(self):
        pass

    def lr(self):
        return self._last_lr[0]


class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (
                self.t - self.warmup_steps) / (
                        self.total_steps - self.warmup_steps)
        return value


class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t - self.warmup_steps) / (
                self.total_steps - self.warmup_steps) * math.pi
        value = self.min_value + (self.max_value - self.min_value) * (
                np.cos(phase) + 1.) / 2.0
        return value


class PolyScheduler(SchedulerBase):
    def __init__(self, poly_order=-0.5, *args, **kwargs):
        super(PolyScheduler, self).__init__(*args, **kwargs)
        self.poly_order = poly_order
        assert poly_order <= 0, "Please check poly_order<=0 so that the scheduler decreases with steps"

    def step_func(self):
        value = self.min_value + (self.max_value - self.min_value) * (
                self.t - self.warmup_steps) ** self.poly_order
        return value
