import yaml

from yaml.representer import SafeRepresenter
from yaml.emitter import Emitter
from yaml.serializer import Serializer
from yaml.resolver import Resolver



class YamlRepresenter(SafeRepresenter):

    def represent_object(self, data):
        return self.represent_str(str(data))

YamlRepresenter.add_multi_representer(object, YamlRepresenter.represent_object)

class YamlDumper(Emitter, Serializer, YamlRepresenter, Resolver):

    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):
        Emitter.__init__(self, stream, canonical=canonical,
                indent=indent, width=width,
                allow_unicode=allow_unicode, line_break=line_break)
        Serializer.__init__(self, encoding=encoding,
                explicit_start=explicit_start, explicit_end=explicit_end,
                version=version, tags=tags)
        YamlRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)
        Resolver.__init__(self)



def load(f):
    return yaml.load(f, Loader=yaml.Loader)

def dump(obj, f):
    return yaml.dump(obj, f, default_flow_style=False, Dumper=YamlDumper)
