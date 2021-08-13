all_plugins = []

def load_all_plugins():
    from .standard_plugin import StandardPlugin
    from .torch_plugin import TorchPlugin

    standard_plugin = StandardPlugin()
    all_plugins.append(standard_plugin)
    
    torch_plugin = TorchPlugin()
    all_plugins.append(torch_plugin)

class ApiTemplate:
    pass

def construct_api():
    obj = ApiTemplate()
    for plugin in all_plugins:
        plugin.extend_api(obj)

    return obj

def experiment_created(experiment):
    for plugin in all_plugins:
        plugin.experiment_created(experiment)

def init_experiment(experiment, params):
    for plugin in all_plugins:
        plugin.before_experiment(experiment, params)

def cleanup():
    for plugin in all_plugins:
        plugin.after_experiment()


class PystematicPlugin:

    def experiment_created(self, experiment):
        """Gives the plugin a chance to modify an experiment when it is created
        """
        pass

    def extend_api(self, api_object):
        """Gives the plugin a chance to modify the pystematic API.
        """
        pass

    def before_experiment(self, experiment, params):
        """Called before the main function of the experiment is executed.

        Args:
            experiment (Experiment): A handle to the experiment object.
            params (dict): Contains the values assigned to the parameters of the experiment.
        """
        pass

    def after_experiment(self):
        """Called after the experiment main function has returned. 
        """
        pass