import torch
import logging
from .recording import Recorder
logger = logging.getLogger('pystematic_torchcontext')



class TorchContext:
    """A context object is like a big container that holds all pytorch related
    objects you need. Its main use is to allow a pytorch session so transition
    seamlessly between different modes (e.g. distributed, cuda) based on
    experiment parameters. It does this by transparently transforming some
    object that you add. For example, when running in distributed mode, all
    pytorch models added to this context will be automatically wrapped in 
    torch's :obj:`DistributedDataParallel`.

    The methods :meth:`state_dict` and :meth:`load_state_dict` provides a single
    point of entry to the state of the entire session (provided that all objects
    are registered with it).

    """

    def cuda(self):
        raise NotImplementedError()

    def cpu(self):
        raise NotImplementedError()

    def ddp(self):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()

    def autotransform(self):
        from pystematic.torch import params
        if params["checkpoint"]:
            logger.info(f"Loading checkpoint '{params['checkpoint']}'.")
            with open(params["checkpoint"], "rb") as f:
                self.load_state_dict(torch.load(f, map_location="cpu"))

        if params["cuda"]:
            self.cuda()

        if params["distributed"]:
            self.ddp()
 
    def _wrap_value(self, value):
        if isinstance(value, (list, tuple)):
            new_value = ContextList()
            for val in value:
                new_value.append(val)

            return new_value
        elif isinstance(value, dict):
            new_value = ContextDict()
            for key, val in value.items():
                new_value[key] = val
            
            return new_value

        return value

    def _to_cuda(self, item):
        if callable(getattr(item, "cuda", None)):
            return item.cuda()
        elif isinstance(item, torch.optim.Optimizer):
            return self._move_to_device(item, f"cuda:{torch.cuda.current_device()}")
        return item

    def _to_cpu(self, item):
        if callable(getattr(item, "cpu", None)):
            return item.cpu()
        elif isinstance(item, torch.optim.Optimizer):
            return self._move_to_device(item, "cpu")
        return item

    def _to_ddp(self, name, item):
        if isinstance(item, TorchContext):
            item.ddp()

        elif isinstance(item, torch.nn.Module):
            if any([p.requires_grad for p in item.parameters()]):
                logger.debug(f"Converting to distributed for model '{name}'.")

                item = torch.nn.parallel.DistributedDataParallel(
                    module=item,
                    device_ids=[torch.cuda.current_device()]
                )

        elif isinstance(item, Recorder):
            if torch.distributed.get_rank() != 0: # Only rank zero may log stats
                item.silence()
                logger.debug(f"Silencing recorder '{name}' in rank '{torch.distributed.get_rank()}'.")

        return item
    
    def _get_state_dict(self, item):
        supported_types = (int, float, complex, str)
        
        if callable(getattr(item, "state_dict", None)):
            if isinstance(item, torch.nn.parallel.DistributedDataParallel):
                return item.module.state_dict()
            else:
                return item.state_dict()

        elif isinstance(item, supported_types):
            return {
                "native_value": item
            }
        else:
            logger.debug(f"Cannot checkpoint object of type '{type(item)}'")

        return None

    def _set_state_dict(self, item, state_dict):
        
        supported_types = (int, float, complex, str)
        if callable(getattr(item, "load_state_dict", None)):
            if isinstance(item, torch.nn.parallel.DistributedDataParallel):
                item.module.load_state_dict(self._move_to_same_device_as(state_dict, item.module))
            else:
                item.load_state_dict(self._move_to_same_device_as(state_dict, item))

        elif isinstance(item, supported_types):
            return state_dict["native_value"] if "native_value" in state_dict else state_dict["count"]
            
        else:
            logger.debug(f"Cannot checkpoint object of type '{type(item)}'.")

        return item

    def _move_to_same_device_as(self, to_move, target):
        if hasattr(target, "device"):
            return self._move_to_device(to_move, target.device)

        elif callable(getattr(target, "parameters", None)):
            try:
                return self._move_to_device(to_move, next(target.parameters()).device)
            except StopIteration:
                pass

        return to_move

    def _move_to_device(self, obj, device):
        if isinstance(obj, dict):
            res = {}
            for name, value in obj.items():
                res[name] = self._move_to_device(value, device)

        elif isinstance(obj, (list, tuple)):
            res = []
            for i in range(len(obj)):
                res.append(self._move_to_device(obj[i], device))

        elif callable(getattr(obj, "to", None)):
            res = obj.to(device=device)
        elif isinstance(obj, torch.optim.Optimizer):
            obj.load_state_dict(self._move_to_device(obj.state_dict(), device))
            res = obj
        else:
            res = obj
            # raise Exception(f"Unsupported object type '{type(obj)}'")

        return res


class ContextObject(TorchContext):
    
    def __init__(self):
        object.__setattr__(self, "_items", {})

    def __getattr__(self, name):
        if name in self._items:
            return self._items[name]

        raise AttributeError(f"TorchContext does not have an attribute named '{name}'.")

    def __setattr__(self, name, value):
        self._items[name] = self._wrap_value(value)

    def has(self, name : str):
        return name in self._items

    def cuda(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cuda(item)

        return self
  
    def cpu(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cpu(item)
        
        return self

    def ddp(self):
        assert torch.distributed.is_initialized(), "You must initialize a distributed runtime before calling ddp."
        
        for name, item in self._items.items():
            self._items[name] = self._to_ddp(name, item)
        
        return self
        
    def state_dict(self) -> dict:
        dict_with_state = {}

        for name, item in self._items.items():
            dict_with_state[name] = self._get_state_dict(item)

        return dict_with_state

    def load_state_dict(self, state : dict) -> None:

        for name, item_state in state.items():
            if name in self._items:
                self._items[name] = self._set_state_dict(self._items[name], item_state)
            

class ContextDict(TorchContext):
    
    def __init__(self):
        object.__setattr__(self, "_items", {})

    def __getitem__(self, name):
        return self._items[name]

    def __setitem__(self, name, value):
        self._items[name] = self._wrap_value(value)

    def items(self):
        return {key:item for key, item in self._items}

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def cuda(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cuda(item)

        return self
  
    def cpu(self):
        for name, item in self._items.items():
            self._items[name] = self._to_cpu(item)
        
        return self

    def ddp(self):
        assert torch.distributed.is_initialized(), "You must initialize a distributed runtime before calling ddp."
        
        for name, item in self._items.items():
            self._items[name] = self._to_ddp(name, item)
        
        return self

    def state_dict(self) -> dict:
        dict_with_state = {}

        for name, item in self._items.items():
            dict_with_state[name] = self._get_state_dict(item)

        return dict_with_state

    def load_state_dict(self, state : dict) -> None:

        for name, item in state.items():
            if name in self._items:
                self._items[name] = self._set_state_dict(self._items[name], item)
            

class ContextList(TorchContext):
    
    def __init__(self):
        object.__setattr__(self, "_items", [])

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, name, value):
        self._items[name] = self._wrap_value(value)

    def insert(self, index, value):
        self._items.insert(index, self._wrap_value(value))

    def append(self, value):
        self._items.append(self._wrap_value(value))

    def __len__(self):
        return len(self._items)
    
    def cuda(self):
        for i, item in enumerate(self._items):
            self._items[i] = self._to_cuda(item)
        
        return self
  
    def cpu(self):
        for i, item in enumerate(self._items):
            self._items[i] = self._to_cpu(item)
        
        return self

    def ddp(self):
        assert torch.distributed.is_initialized(), "You must initialize a distributed runtime before calling ddp."
        
        for i, item in enumerate(self._items):
            self._items[i] = self._to_ddp(item)
        
        return self
    
    def state_dict(self) -> dict:
        """Returns the combined state_dict of all contained items

        Returns:
            dict: A dict that maps names to state_dicts
        """
        list_with_state = {}

        for i, item in enumerate(self._items):
            list_with_state.append(self._get_state_dict(item))

        return list_with_state

    def load_state_dict(self, state : dict) -> None:

        for i, item in enumerate(self.state):
            self._items[i] = self._set_state_dict(self._items[i], item)
       