import click

def invoke_command_with_parsed_args(command, args_dict):
    extra = {}
    for key, value in command.context_settings.items():
        if key not in extra:
            extra[key] = value

    ctx = click.Context(command, **extra)

    for param in command.get_params(ctx):
        value, _ = param.handle_parse_result(ctx, args_dict, None)

    with ctx as ctx:
        command.invoke(ctx)
