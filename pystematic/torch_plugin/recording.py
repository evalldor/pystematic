import pathlib

import matplotlib.figure
import numpy as np
import tensorboardX
import torch
from PIL import Image

class Recorder:
    """Used for recording metrics during training and evaluation.
    
    The recorder has an internal counter :attr:`count` that is recorded
    together with all values. The count typically represents the
    'global_step' during training. Remember to increment the counter
    appropriately.

    Each recorded value is also associated with a ``tag`` that uniquely
    determines which time series the value should be recorded to. The tag
    can use slashes ('/') to build hierarchies. E.g. ``train/loss``,
    ``test/loss`` etc.
    """

    def __init__(self, output_dir=None, tensorboard=True, file=True, console=False):
        """
        
        Args:
            output_dir (str, optional): The output directory store data in.
                Defaults to :attr:`pystematic.output_dir`.
            tensorboard (bool, optional): If the recorder should write tensorboard 
                logs. Defaults to True.
            file (bool, optional): If the recorder should write to plain files. 
                Defaults to True.
            console (bool, optional): If the recorder should write to stdout. 
                Defaults to False.
        """
        self._counter = 0

        if output_dir is None:
            import pystematic as pst
            output_dir = pst.output_dir
            
        self._output_dir = output_dir
        self._recording_backends = []
        
        if tensorboard:
            self._recording_backends.append(TensorboardBackend(self._output_dir))
        
        if file:
            self._recording_backends.append(FileBackend(self._output_dir))

        if console:
            self._recording_backends.append(ConsoleBackend())

    def silence(self):
        """A recorder may be silenced when running distributed training, in
        order to restrict logging to the master process.
        
        This method is called by the pytorch context.
        """
        self._recording_backends = []

    @property
    def count(self) -> int:
        """Internal counter that represents the x-axis when logging data.
        """
        return self._counter

    @count.setter
    def count(self, value):
        self._counter = value
    
    def step(self):
        """Increases :attr:`count` by 1."""

        self._counter += 1
        
        for backend in self._recording_backends:
            backend.step()

    def params(self, params_dict):
        for backend in self._recording_backends:
            backend.params(params_dict)

    def scalar(self, tag, scalar):
        """Logs a scalar value.

        Args:
            tag (str): A string that represents the 'name' of the scalar.
            scalar (float): The value of the scalar.
        """
        if torch.is_tensor(scalar):
            scalar = scalar.cpu().item()
        
        for backend in self._recording_backends:
            backend.scalar(tag, scalar, self.count)
            
    def figure(self, tag, fig):
        """Logs a matplotlib figure

        Args:
            tag (str): A string tag
            fig (Figure): A matplotlib figure
        """

        if not isinstance(fig, matplotlib.figure.Figure):
            raise ValueError(f"Figure must be an instance of 'matplotlib.figure.Figure', got '{type(fig)}'.")

        for backend in self._recording_backends:
            backend.figure(tag, fig, self.count)

    def image(self, tag, image):
        """Logs an image

        Args:
            tag (str): The tag
            image (PIL.Image, np.ndarray, torch.tensor): The image
        """

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        for backend in self._recording_backends:
            backend.image(tag, image, self.count)

    def state_dict(self):
        return {
            "count": self.count
        }
    
    def load_state_dict(self, state):
        self._counter = state["count"]


class RecorderBackend:

    def step(self):
        """Called every time the counter in the logger is changed. Can be used
        to flush buffers etc.
        """
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


class FileBackend(RecorderBackend):

    def __init__(self, output_dir):
        self._output_dir = output_dir

    def scalar(self, tag, scalar, counter):
        scalars_file = pathlib.Path(self._output_dir).joinpath("scalars.csv")
        
        if not scalars_file.exists():
            with scalars_file.open("w") as f:
                f.write("step, tag, value\n")

        with scalars_file.open("a") as f:
            f.write(f"{counter}, {tag}, {scalar}\n")


    def figure(self, tag, fig, counter):
        figures_folder = pathlib.Path(self._output_dir).joinpath("figures")

        figures_folder.mkdir(exist_ok=True, parents=True)
        fig_path = figures_folder.joinpath(f"{tag.replace('/', '.')}-{counter}.jpg")

        fig.savefig(fig_path)
        

    def image(self, tag, image, counter):
        img_folder = pathlib.Path(self._output_dir).joinpath("images")
        img_folder.mkdir(exist_ok=True, parents=True)
        img_path = img_folder.joinpath(f"{tag.replace('/', '.')}-{counter}.jpg")

        if isinstance(image, np.ndarray):
            
            if len(image.squeeze().shape()) < 2 or len(image.squeeze().shape()) > 3:
                raise ValueError(f"Invalid image shape '{image.shape}' must have 2 or 3 non-singleton dimensions.")
            
            image = image.squeeze()

            if len(image.shape) == 3:
                image = image.transpose((1, 2, 0)) # CHW -> HWC
            
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError(f"Image is of unsupported type: '{type(image)}'.")
        
        image.save(img_path)
