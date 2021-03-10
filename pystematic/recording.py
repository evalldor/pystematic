import torch
import tensorboardX



class Recorder:
    """Used for recording metrics during training and evaluation."""

    def __init__(self, output_dir=None, tensorboard=True, console=False):
        
        self._counter = 0

        self._output_dir = output_dir
        self._recording_backends = []
        
        if tensorboard:
            self._recording_backends.append(TensorboardBackend(self._output_dir))
        
        if console:
            self._recording_backends.append(ConsoleBackend())

    def set_output_dir(self, output_dir):
        """This method is called by the pytorch context."""
        self._output_dir = output_dir

        for backend in self._recording_backends:
            backend.set_output_dir(self._output_dir)

    def silence(self):
        """A recorder may be silenced when running distributed training, in
        order to restrict logging to the master process.
        
        This method is called by the pytorch context.
        """
        self._recording_backends = []

    @property
    def count(self):
        return self._counter

    @count.setter
    def count(self, value):
        self._counter = value
    
    def step(self):
        """Increases the counter by 1."""

        self._counter += 1
        
        for backend in self._recording_backends:
            backend.step()

    def params(self, params_dict):
        for backend in self._recording_backends:
            backend.params(params_dict)

    def scalar(self, tag, scalar):
        if torch.is_tensor(scalar):
            scalar = scalar.cpu().item()
        
        for backend in self._recording_backends:
            backend.scalar(tag, scalar, self.count)
            
    def figure(self, tag, fig):
        for backend in self._recording_backends:
            backend.figure(tag, fig, self.count)

    def state_dict(self):
        return {
            "count": self.count
        }
    
    def load_state_dict(self, state):
        self._counter = state["count"]


class RecorderBackend:

    def set_output_dir(self, output_dir):
        pass

    def step(self):
        pass
    
    def params(self, params_dict):
        pass

    def scalar(self, tag, scalar, counter):
        pass

    def figure(self, tag, fig, counter):
        pass

    def image(self, tag, image, counter):
        pass


class TensorboardBackend(RecorderBackend):

    shared_tensorboard_logger = None

    def __init__(self, output_dir):
        self._output_dir = output_dir

    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    def _get_tb_logger(self):
        if TensorboardBackend.shared_tensorboard_logger is None:
            TensorboardBackend.shared_tensorboard_logger = tensorboardX.SummaryWriter(self._output_dir)

        return TensorboardBackend.shared_tensorboard_logger

    def params(self, params_dict):
        self._get_tb_logger().add_hparams(params_dict, {})

    def scalar(self, tag, scalar, counter):
        self._get_tb_logger().add_scalar(tag, scalar, counter)

    def figure(self, tag, fig, counter):
        self._get_tb_logger().add_figure(tag, fig, counter, close=True)

    def image(self, tag, image, counter):
        self._get_tb_logger().add_image(tag, image, counter)


class ConsoleBackend(RecorderBackend):

    def __init__(self):
        self._scalars = {}

    def step(self):
        print("\n".join([f"{name}: {value}" for name, value in self._scalars.items()]))
        print("\n")
        self._scalars = {}

    def scalar(self, tag, scalar, counter):
        self._scalars["counter"] = counter
        self._scalars[tag] = scalar

