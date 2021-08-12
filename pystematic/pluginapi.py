all_plugins = []
api_objects = []

def load_all_plugins():
    from .standard_plugin import StandardPlugin
    from .torch_plugin import TorchPlugin

    standard_plugin = StandardPlugin()
    all_plugins.append(standard_plugin)
    
    torch_plugin = TorchPlugin()
    all_plugins.append(torch_plugin)

class Tmp:
    pass

def construct_api_extension():
    obj = Tmp()
    for plugin in all_plugins:
        api_object = plugin.get_api_extension()
        namespace = plugin.get_api_namespace()

        if namespace is None:
            object_to_extend = obj
        else:
            setattr(obj, namespace, Tmp())
            object_to_extend = getattr(obj, namespace)

        for name in dir(api_object):
            if not name.startswith("_"):
                setattr(object_to_extend, name, getattr(api_object, name))
    
    return obj

def experiment_created(experiment):
    for plugin in all_plugins:
        plugin.experiment_created(experiment)

def init_experiment(experiment, params):
    for api_object in api_objects:
        api_object._init_experiment_(experiment, params)

def cleanup():
    for api_object in api_objects:
        api_object._cleanup_()


class PystematicPlugin:

    def experiment_created(self, experiment):
        """Gives the plugin a chance to modify an experiment when it is created
        """
        pass

    def get_api_extension(self):
        """Returns an ApiExtension object that will be used to extend the public
        API under the `pystematic` namespace.
        """
        pass

    def get_api_namespace(self):
        pass
