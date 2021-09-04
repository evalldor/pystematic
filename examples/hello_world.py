import pystematic as ps

@ps.parameter(
    name="name",
    type=str,
    help="The name to greet.",
    required=True
)
@ps.experiment
def hello_world(params):
    print(f"Hello {params['name']}!")


if __name__ == "__main__":
    hello_world.cli()
