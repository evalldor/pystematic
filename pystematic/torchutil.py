import abc
import math
from random import sample

import numpy as np
import torch
import tensorboardX
import tqdm

class DistributedSampler(torch.utils.data.distributed.Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if not torch.distributed.is_initialized():
            raise Exception("Distributed sampler can only be used in a distributed environment.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).cuda()

            torch.distributed.broadcast(indices, 0)
            indices = indices.cpu().tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples, "{} != {}".format(len(indices), self.num_samples)

        return iter(indices)

    def __len__(self):
        return self.num_samples


class Logger:
    """Used for logging metrics during training and evaluation."""

    def __init__(self, log_dir=None, global_step_getter=None, epoch_getter=None, dummy=False):
        """A dummy logger may be created when running distributed training, in
        order to restrict logging to the master process."""

        self._scalars = {}
        self._figures = {}
        self._images = {}
        self._dummy = dummy
        self._global_step_getter = global_step_getter
        self._epoch_getter = epoch_getter

        self._log_dir = log_dir
        self._tb_logger = None #SummaryWriter(log_dir)
        #self._tb_logger.logdir

        self._log_to_tb = True
        self._log_to_console = False

    def get_tb_logger(self):
        if self._tb_logger is None:
            self._tb_logger = tensorboardX.SummaryWriter(self._log_dir)
        
        return self._tb_logger

    def log_to_tensorboard(self, log_to_tb):
        self._log_to_tb = log_to_tb

    def log_to_console(self, log_to_console):
        self._log_to_console = log_to_console

    def log_dir(self):
        return self._log_dir

    def scalar(self, tag, scalar):
        if torch.is_tensor(scalar):
            scalar = scalar.cpu().item()

        self._scalars[tag] = scalar

    def image_grid(self, tag, images):
        
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        
        if len(images.shape) > 5:
            raise Exception("images.shape must be <= 5 (nrows, ncols, h, w, c)")

        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        if len(images.shape) == 4:
            images = np.expand_dims(images, axis=0)

        nrows = images.shape[0]
        ncols = images.shape[1]
        
        all_rows = []

        for row in range(nrows):
            curr_row = []
            for col in range(ncols):
                curr_row.append(_img_norm(images[row, col]))

            all_rows.append(np.concatenate(curr_row, axis=2))
       

        img_grid = np.concatenate(all_rows, axis=1)
        
        self._images[tag] = img_grid

    def figure(self, tag, fig):
        self._figures[tag] = fig

    def clear(self):
        self._scalars = {}
        self._figures = {}
        self._images = {}

    def commit(self):
        if not self._dummy:
            global_step = self._global_step_getter()
            epoch = self._epoch_getter()

            if self._log_to_tb:
                for name, value in self._scalars.items():
                    self.get_tb_logger().add_scalar(name, value, global_step)

                for name, value in self._figures.items():
                    self.get_tb_logger().add_figure(name, value, global_step, close=True)

                for name, value in self._images.items():
                    self.get_tb_logger().add_image(name, value, global_step)
                
                self.get_tb_logger().flush()
            
            if self._log_to_console:
                log_strings = [
                    "epoch: {}".format(epoch),
                    "global step: {}".format(global_step)
                ]

                for name, value in self._scalars.items():
                    log_strings.append("{}: {:.5f}".format(name, value))

                print("\n".join(log_strings))
                print("\n")

        self.clear()


def _img_norm(img):
    max_val = np.amax(img)
    min_val = np.amin(img)

    img = (img - min_val) / (max_val - min_val)

    return img


class Looper(abc.ABC):
    """A utility class to help with looping over a dataset. You are expected to
    subclass this class and override the desired methods.
    `loop_step` is called once for each item yielded by the iterator.
    
    `before_loop` and `after_loop` are called before and after the iteration begins/ends.

    A single iteration through the while iterator is done by calling the `loop_once` method.
    """
    @abc.abstractmethod
    def get_iterator(self):
        """Must return the iterator that should be looped over."""
        pass

    def before_loop(self):
        pass

    def after_loop(self):
        pass

    def loop_step(self, item, current_step, total_num_steps):
        pass

    def loop_once(self):
        #Only show progressbar for the master process
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            self._loop_once_without_progessbar()
        else:
            self._loop_once_with_progessbar()

    def _loop_once_with_progessbar(self):
        dataloader = self.get_iterator()

        self.before_loop()

        with tqdm.tqdm(dataloader, leave=True) as progress_bar:
            
            for curr_step, item in enumerate(progress_bar, 1):
                message = self.loop_step(item, curr_step, len(progress_bar))
                
                if message is not None:
                    progress_bar.set_description(message)

        self.after_loop()
    
    def _loop_once_without_progessbar(self):

        dataloader = self.get_iterator()

        self.before_loop()

        for curr_step, item in enumerate(dataloader, 1):
            self.loop_step(item, curr_step, len(dataloader))

        self.after_loop()


def create_sampler(dataset, shuffle=True, seed=0):
    """Returns a DistributedSampler if running in distributed mode, otherwise a normal sampler

    Args:
        dataset (torch.utils.data.Dataset): The dataset the sampler will work on.
        shuffle (bool): If the sampler should be random or not.
    """

    if torch.distributed.is_initialized():
        return BetterDistributedSampler(
            dataset=dataset, 
            shuffle=shuffle,
            seed=seed
        )

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.utils.data.RandomSampler(data_source=dataset, generator=g)
    
    return torch.utils.data.SequentialSampler(data_source=dataset)


class BetterDistributedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, shuffle=True, seed=0):
        super().__init__(dataset, shuffle=shuffle, seed=seed)
        self.epoch = 0

    def __iter__(self):
        self.set_epoch(self.epoch+1)
        return super().__iter__()


class BetterDataLoader(torch.utils.data.DataLoader):
    """ Implements a dataloader data works consistently in both distributed and nondistributed runtimes
    """
    def __init__(self, dataset, shuffle=False, random_seed=0, **kwargs):
        super().__init__(dataset, sampler=create_sampler(dataset, shuffle, random_seed), **kwargs)
        