import sys
import pprint
import click

@click.group()
def global_entrypoint():
    """pystematic global entrypoint. Below is a list of all registered
    experiments. Append the name of the experiment you would like to run to the
    commandline you invoked to run this script.
    """
    """All experiments are registered with this Click group. In your main
    script, simply call this function to access the CLI for all registered
    experiments.
    """
    pass


def get_all_registered_experiments():
    pass

def invoke_command_with_parsed_args(func, args_dict):
    
    # if experiment_name in global_entrypoint.commands:
    #     command = global_entrypoint.commands[experiment_name]

    extra = {}
    for key, value in func.command.context_settings.items():
        if key not in extra:
            extra[key] = value

    ctx = click.Context(func.command, **extra)

    for param in func.command.get_params(ctx):
        value, _ = param.handle_parse_result(ctx, args_dict, None)

    with ctx as ctx:
        func.command.invoke(ctx)

    # else:
    #     raise Exception(f"Could not find an experiment named '{experiment_name}'")