import pystematic as ps

@ps.parameter(
    name="ungrouped_param",
    type=str,
    help="An ungrouped param",
    required=True
)
@ps.param_group("A param group",
    ps.parameter(
        name="param_a_1",
        type=str,
        help="A param in group A"
    ),
    ps.parameter(
        name="param_a_2",
        type=str,
        help="Another param in group A"
    )
)
@ps.param_group("Another param group",
    ps.parameter(
        name="param_b_1",
        type=str,
        help="A param in group Another "
    ),
    ps.parameter(
        name="param_b_2",
        type=str,
        help="Another param in group Another"
    )
)
@ps.experiment
def param_groups(params):
    print(f"Hello {params['name']}!")


if __name__ == "__main__":
    param_groups.cli()
