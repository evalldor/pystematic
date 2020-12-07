import click
import click.testing

import pystematic as ps

@ps.pytorch_experiment
@click.option("--test/--notest",
    default=False,
    help="A test flag.",
    show_default=True,
    show_envvar=True
)
def a_command(ctx):
    print(ctx.config())
    assert ctx.config("output_dir") == "test"
    

def test_cli():
    runner = click.testing.CliRunner()
    result = runner.invoke(a_command, ["--help"])
    assert result.exit_code == 0, result.output
    
    result = runner.invoke(a_command, ["--output-dir", "test"])
    assert result.exit_code == 0, result.output

    

if __name__ == "__main__":
    a_command()