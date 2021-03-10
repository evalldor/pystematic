import abc
import math

import torch
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

    def loop_step(self, item):
        pass

    def loop_once(self, progessbar=True):
        #Only show progressbar for the master process
        if not progessbar or (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
            self._loop_once_without_progessbar()
        else:
            self._loop_once_with_progessbar()

    def _loop_once_with_progessbar(self):
        dataloader = self.get_iterator()

        self.before_loop()

        with tqdm.tqdm(dataloader, leave=True) as progress_bar:
            
            for item in progress_bar:
                message = self.loop_step(item)
                
                if message is not None:
                    if isinstance(message, str):
                        progress_bar.set_description(message)
                    elif isinstance(message, dict):
                        progress_bar.set_description(", ".join([f"{name}: {value}" for name, value in message.items()]))


        self.after_loop()
    
    def _loop_once_without_progessbar(self):

        dataloader = self.get_iterator()

        self.before_loop()

        for item in dataloader:
            self.loop_step(item)

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
    """This class extends torch's default DistributedSampler but removes the need
    for manually calling the set_epoch method to reseed the random sampler
    """
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
        