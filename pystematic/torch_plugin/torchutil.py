import math

import torch

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
