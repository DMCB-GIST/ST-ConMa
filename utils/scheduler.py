import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import warnings


def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler based on config

    Args:
        optimizer: optimizer instance
        config: configuration dict with scheduler settings

    Returns:
        scheduler instance
    """
    scheduler_name = config.get('scheduler', 'cosine').lower()

    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('min_lr', 0)
        )

    elif scheduler_name == 'cosine_warmup':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=config.get('epochs', 100),
            cycle_mult=config.get('cycle_mult', 1.0),
            max_lr=config.get('lr', 0.1),
            min_lr=config.get('min_lr', 1e-6),
            warmup_steps=config.get('warmup_epochs', 0),
            gamma=config.get('gamma', 1.0)
        )

    elif scheduler_name == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 0.1),
            total_steps=config.get('total_steps', None),
            epochs=config.get('epochs', None),
            steps_per_epoch=config.get('steps_per_epoch', None),
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 1e4)
        )

    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.1),
            patience=config.get('patience', 10),
            threshold=config.get('threshold', 1e-4),
            min_lr=config.get('min_lr', 0)
        )

    elif scheduler_name == 'linear_warmup':
        return LinearWarmupScheduler(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config.get('epochs', 100)
        )

    elif scheduler_name == 'linear_warmup_cosine':
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            max_epochs=config.get('epochs', 100),
            warmup_start_lr=config.get('warmup_start_lr', 1e-8),
            eta_min=config.get('min_lr', 0)
        )

    elif scheduler_name == 'none' or scheduler_name is None:
        return None

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear learning rate warmup followed by constant learning rate.
    Useful for stabilizing training at the beginning.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs=None, last_epoch=-1):
        """
        Args:
            optimizer: wrapped optimizer
            warmup_epochs: number of warmup epochs
            total_epochs: total number of epochs (optional)
            last_epoch: the index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # Constant after warmup
            return self.base_lrs


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    Widely used in Vision Transformers (ViT), BERT, etc.
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs,
                 warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        """
        Args:
            optimizer: wrapped optimizer
            warmup_epochs: number of warmup epochs
            max_epochs: maximum number of training epochs
            warmup_start_lr: learning rate to start the warmup from
            eta_min: minimum learning rate
            last_epoch: the index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) *
                    self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    Combines the benefits of warmup, cosine annealing, and restarts.
    """

    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.,
                 max_lr=0.1, min_lr=1e-6, warmup_steps=0, gamma=1., last_epoch=-1):
        """
        Args:
            optimizer: wrapped optimizer
            first_cycle_steps: number of steps for the first cycle
            cycle_mult: cycle length multiplier after each restart
            max_lr: maximum learning rate
            min_lr: minimum learning rate
            warmup_steps: number of warmup steps
            gamma: learning rate decay factor after each restart
            last_epoch: the index of last epoch
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # Initialize learning rate
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Warmup
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            return [base_lr + (self.max_lr - base_lr) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up learning rate, then switch to another scheduler.
    Wrapper scheduler that adds warmup to any scheduler.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """
        Args:
            optimizer: wrapped optimizer
            multiplier: target learning rate = base lr * multiplier
            total_epoch: number of warmup epochs
            after_scheduler: scheduler to use after warmup
        """
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def create_scheduler_with_warmup(optimizer, base_scheduler_config, warmup_epochs=5):
    """
    Create a scheduler with warmup wrapper.

    Args:
        optimizer: optimizer instance
        base_scheduler_config: config for the base scheduler after warmup
        warmup_epochs: number of warmup epochs

    Returns:
        scheduler with warmup
    """
    base_scheduler = get_scheduler(optimizer, base_scheduler_config)

    if warmup_epochs > 0 and base_scheduler is not None:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=warmup_epochs,
            after_scheduler=base_scheduler
        )
        return scheduler
    else:
        return base_scheduler


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, eta_min=0.0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr and eta_min, after a warmup period.

    This is commonly used in NLP models like BERT, RoBERTa, etc.

    Returns:
        scheduler instance
    """
    # Get base learning rates from optimizer
    base_lrs = [group['lr'] for group in optimizer.param_groups]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup: 0 -> 1.0
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing with min_lr support
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

        # We want: lr = eta_min + (base_lr - eta_min) * cosine_decay
        # LambdaLR computes: lr = base_lr * lr_lambda(step)
        # Therefore: lr_lambda = (eta_min + (base_lr - eta_min) * cosine_decay) / base_lr
        #                      = eta_min/base_lr + (1 - eta_min/base_lr) * cosine_decay
        min_lr_ratio = eta_min / base_lrs[0] if base_lrs[0] > 0 else 0.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr
    to 0, after a warmup period.

    Returns:
        scheduler instance
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)