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

    def __init__(self, log_dir=None, dummy=False):
        """A dummy logger may be created when running distributed training, in
        order to restrict logging to the master process."""

        self._scalars = {}
        self._figures = {}
        self._images = {}
        self._global_step = -1
        self._epoch = -1
        self._dummy = dummy

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

    def global_step(self, global_step):
        self._global_step = global_step

    def epoch(self, epoch):
        self._epoch = epoch

    def clear(self):
        self._scalars = {}
        self._figures = {}
        self._images = {}
        self._global_step = -1
        self._epoch = -1

    def commit(self):
        if not self._dummy:

            if self._log_to_tb:
                for name, value in self._scalars.items():
                    self.get_tb_logger().add_scalar(name, value, self._global_step)

                for name, value in self._figures.items():
                    self.get_tb_logger().add_figure(name, value, self._global_step, close=True)

                for name, value in self._images.items():
                    self.get_tb_logger().add_image(name, value, self._global_step)
                
                self.get_tb_logger().flush()
            
            if self._log_to_console:
                log_strings = [
                    "epoch: {}".format(self._epoch),
                    "global step: {}".format(self._global_step)
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

    @abc.abstractmethod
    def get_iterator(self):
        pass

    def before(self):
        pass

    def after(self):
        pass

    def step(self, item, current_step, total_num_steps):
        pass

    def loop_once(self):
        #Only show progressbar for the master process
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            self._loop_once_without_progessbar()
        else:
            self._loop_once_with_progessbar()

    def _loop_once_with_progessbar(self):
        dataloader = self.get_iterator()

        self.before()

        with tqdm.tqdm(dataloader, leave=True) as progress_bar:
            
            for curr_step, item in enumerate(progress_bar, 1):
                message = self.step(item, curr_step, len(progress_bar))
                
                if message is not None:
                    progress_bar.set_description(message)

        self.after()
    
    def _loop_once_without_progessbar(self):

        dataloader = self.get_iterator()

        self.before()

        for curr_step, item in enumerate(dataloader, 1):
            self.step(item, curr_step, len(dataloader))

        self.after()


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
        