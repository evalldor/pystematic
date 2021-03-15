import inspect

import click


class Parameter(click.Option):
    """Extends the default click option with a flag for whether the option is
    allowed to be loaded from a parameters file.
    """

    def __init__(self, *, name, allow_from_params_file=True, **attrs):

        assert not "-" in name, "name cannot contain '-' characters. Use '_' instead!"

        param_decls = f"--{name.lower().replace('_', '-')}"

        super().__init__([param_decls], **attrs)

        self.allow_from_params_file = allow_from_params_file


class Label(click.Option):

    def __init__(self, name):
        super().__init__([], expose_value=False, hidden=True, allow_from_autoenv=False)
        self.label = name


class ExperimentClickCommand(click.Command):

    def __init__(self, name, **attrs):

        super().__init__(name, **attrs)
    
    def format_options(self, ctx, formatter):
        """Writes all the options into the formatter if they exist. Allows for
        `Label` options.
        """

        opts = {
            None: []
        }
        labels = [None]
        current_label = None

        for param in self.get_params(ctx):

            if isinstance(param, Label):
                current_label = param.label
                labels.append(current_label)
                opts[current_label] = []
            else:
                rv = param.get_help_record(ctx)

                if rv is not None:
                    opts[current_label].append(rv)

        
        with formatter.section("Options"):   

            for label in labels:
                if len(opts[label]) > 0:
                    if label is not None:
                        with formatter.section(label):
                            formatter.write_dl(opts[label], col_max=40)
                    else:
                        formatter.write_dl(opts[label], col_max=40)


class ExperimentCollection(click.Group):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_usage(self, ctx, formatter):
        """Writes the usage line into the formatter.

        This is a low-level method called by :meth:`get_usage`.
        """
        pieces = self.collect_usage_pieces(ctx)
        # formatter.write_usage(ctx.command_path, " ".join(pieces))
        formatter.write_usage(ctx.command_path, "[OPTIONS] EXPERIMENT_NAME [PARAMETERS]")
    
    def format_help_text(self, ctx, formatter):
        """Writes the help text to the formatter if it exists."""
        
        formatter.write_paragraph()
        with formatter.indentation():
            help_text = inspect.cleandoc("""Pystematic global entrypoint. Below is a list of all registered
                                            experiments. Append the name of the experiment you would like to
                                            run to the commandline you invoked to run this script.""")
            formatter.write_text(help_text)

    def format_commands(self, ctx, formatter):
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            # What is this, the tool lied about a command.  Ignore it
            if cmd is None:
                continue
            if cmd.hidden:
                continue

            commands.append((subcommand, cmd))

        # allow for 3 times the default spacing
        with formatter.section("Experiments"):
            if len(commands) > 0:
                limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)

                rows = []
                for subcommand, cmd in commands:
                    help = cmd.get_short_help_str(limit)
                    rows.append((subcommand, help))

                formatter.write_dl(rows)

            else:
                formatter.write_text("No experiments defined yet...")



def parameter_decorator(
    name, 
    help=None, 
    type=str, 
    is_flag=False, 
    default=None, 
    required=False,
    allow_from_params_file=True, 
    multiple=False,
    **advanced_args):
    """Decorator for adding parameters to an experiment

    """
    def decorator(experiment):
        kwargs = {
            "name": name,
            "help": help,
            "type": type,
            "is_flag": is_flag,
            "default": default,
            "required": required,
            "allow_from_params_file": allow_from_params_file,
            "multiple": multiple,
            "show_default": True,
            "show_envvar": True
        }

        kwargs.update(advanced_args)

        add_parameter_to_experiment(experiment, Parameter(**kwargs))

        return experiment

    return decorator



@click.group(cls=ExperimentCollection)
def global_entrypoint():
    """This function acts as an entrypoint to all defined experiments. In your
    main script, simply call this function to access the CLI for all registered
    experiments.
    """
    """pystematic global entrypoint. Below is a list of all registered
    experiments. Append the name of the experiment you would like to run to the
    commandline you invoked to run this script.
    """
    pass


def get_current_experiment():
    return click.get_current_context().command._experiment_main_func


def get_experiment_parameters(experiment):
    if isinstance(experiment, click.Command):
        return experiment.params
    elif hasattr(experiment, "__click_params__"):
        return experiment.__click_params__
    
    return []


def add_parameter_to_experiment(experiment, parameter):
    if isinstance(experiment, click.Command):
        experiment.params.append(parameter)
    else:
        if not hasattr(experiment, "__click_params__"):
            experiment.__click_params__ = []
        experiment.__click_params__.append(parameter)

    return experiment


def make_experiment_decorator(default_parameters, experiment_callback):

    def experiment_constructor(experiment_main_func=None, *, name=None, inherit_params=None, defaults=None, **kwargs):

        kwargs["context_settings"] = { # These are passed to the click Command class,
            "default_map": defaults, 
            "show_default": True
        }

        def decorator(experiment_main_func):
            
            experiment_name = name or experiment_main_func.__name__.lower().replace("_", "-")

            def command_initializer(**params):
                
                return experiment_callback(experiment_main_func, params)


            params = []

            if inherit_params is not None:
                params += get_experiment_parameters(inherit_params)

            params += default_parameters

            click_command = ExperimentClickCommand(
                name=experiment_name,
                callback=command_initializer,
                params=params,
                **kwargs
            )

            # Register the command with the global entrypoint.
            global_entrypoint.add_command(click_command)
            
            click_command._experiment_main_func = experiment_main_func
            experiment_main_func.experiment_name = experiment_name
            experiment_main_func._click_command = click_command

            return experiment_main_func

        if callable(experiment_main_func):
            return decorator(experiment_main_func)
        else:
            return decorator

    return experiment_constructor


def invoke_experiment_with_parsed_args(experiment, args_dict):
    
    extra = {}
    
    for key, value in experiment._click_command.context_settings.items():
        if key not in extra:
            extra[key] = value

    ctx = click.Context(experiment._click_command, **extra)

    for param in experiment._click_command.get_params(ctx):
        value, _ = param.handle_parse_result(ctx, args_dict, None)

    with ctx as ctx:
        experiment._click_command.invoke(ctx)