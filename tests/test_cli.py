import click
import click.testing

import pystematic

import pystematic.core as core

def test_parsing():
    params = [
        core.Parameter(
            name="string_param",
            help="A string parameter used for testing"
        ),
        core.Parameter(
            name="int_param",
            help="An int parameter used for testing",
            type=int
        ),
        core.Parameter(
            name="bool_param",
            help="A bool parameter used for testing",
            type=bool
        ),
        core.Parameter(
            name="flag_param",
            help="A flag parameter used for testing",
            is_flag=True
        ),
        core.Parameter(
            name="choice_param",
            help="A choice param used for testing",
            choices=["choice_1", "choice_2", 123]
        ),
        core.Parameter(
            name="multiple_str",
            help="A string parameter that can be added multiple times",
            multiple=True
        ),
        core.Parameter(
            name="multiple_choices",
            help="A choice parameter that can be added multiple times",
            choices=["choice_1", "choice_2", 123],
            multiple=True
        )
    ]

    parser = core._construct_argparser(params)
    
    try:
        print("Output formatting preview")
        print(parser.parse_args(["-h"]))
        print()
    except SystemExit:
        pass

