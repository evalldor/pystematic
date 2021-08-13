import argparse
import pystematic.parametric as parametric
import pytest

def test_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pos_1", 
        help="A positional arg",
        nargs="*"
    )
    parser.add_argument(
        "pos_2", 
        help="A positional arg 2",
        nargs="*"
    )
    parser.add_argument(
        "pos_3", 
        nargs=3
    )
    parser.add_argument(
        "pos_4", 
        help="A positional arg 2",
        nargs="+"
    )
    parser.add_argument(
        "--string-param", 
        help="A string parameter used for testing"
    )
    parser.add_argument(
        "--int-param", 
        help="An int parameter used for testing",
        type=int
    )
    # parser.add_argument(
    #     "--bool-param", 
    #     help="A bool parameter used for testing",
    #     action=argparse.BooleanOptionalAction
    # )
    parser.add_argument(
        "--flag-param", 
        help="A flag parameter used for testing",
        nargs="?"
    )
    parser.add_argument(
        "--choice-param", 
        help="A choice parameter used for testing",
        choices=["choice_1", "choice_2", 123]
    )
    parser.add_argument(
        "--multiple-str", 
        help="A string parameter that can be added multiple times",
        nargs="+"
    )
    parser.add_argument(
        "--multiple-choices", 
        help="A choice parameter that can be added multiple times",
        nargs="+",
        choices=["choice_1", "choice_2", 123]
    )

    # parser.print_help()

    parser.parse_args(["woot", "asd", "odn", "fpo", "sdnf", "--multiple-str", "asd", "asdasd", "--flag-param", "1"])
    res = parser.parse_known_args(["--multiple-str", "asd", "--asdasd", "asdasd", "--flag-param", "1", "woot", "asd", "odn", "fpo", "sdnf"])
    parser.parse_args(["--multiple-str", "asd", "asdasd", "--flag-param", "1", "woot", "asd", "odn", "fpo", "sdnf", "--int-param", "1"])

    # print(res)

def test_parametric():
    params = parametric.ParameterManager()
    
    params.add_param(
        name="pos_1",
        help="A positional arg",
        cli_positional=True
    )
    params.add_param(
        name="pos_2", 
        nargs="*",
        help="A positional arg 2",
        cli_positional=True
    )
    params.add_param(
        name="pos_3", 
        nargs=3,
        help=None,
        cli_positional=True
    )
    params.add_param(
        name="pos_4", 
        nargs="+",
        help="A positional arg 4",
        cli_positional=True
    )

    params.add_param(
        name="string_param",
        flags=["--string-param"],
        help="A string parameter used for testing.",
        default=lambda: "hello"
    )

    params.add_param(
        name="int_param",
        flags=["--int-param"], 
        help="An int parameter used for testing.",
        type=int,
        default=18,
        required=True
    )

    params.add_param(
        name="float_param",
        flags=["--float-param"], 
        help="A float parameter used for testing.",
        type=float
    )

    params.add_param(
        name="flag_param",
        flags=["-f", "--flag-param"],
        help="A flag parameter used for testing.",
        nargs="?"
    )

    params.add_param(
        name="pure_flag",
        flags=["-p", "--pure-flag"],
        help="A flag parameter with no args.",
        nargs=0
    )

    params.add_param(
        name="multiple_str",
        flags=["--multiple-str"],
        help="A string parameter that can be added multiple times.",
        nargs="*",
        default=["hej"]
    )

    

    res = params.from_cli(["--multiple-str", "asd", "asdasd", "--", "woot", "odn", "fpo", "sdnf", "--flag-param"])
    assert res["multiple_str"] == ["asd", "asdasd"]
    assert res["pos_1"] == "woot"
    assert res["pos_2"] == []
    assert res["pos_3"] == ["odn", "fpo", "sdnf"]
    assert res["pos_4"] == ["--flag-param"]

    res = params.from_cli(["--multiple-str", "asd", "asdasd", "-p", "woot", "--", "odn", "fpo", "sdnf", "--flag-param"])
    assert res["multiple_str"] == ["asd", "asdasd"]
    assert res["pos_1"] == "woot"
    assert res["pos_2"] == []
    assert res["pos_3"] == ["odn", "fpo", "sdnf"]
    assert res["pos_4"] == ["--flag-param"]

    res = params.from_cli(["woot", "odn", "fpo", "sdnf", "asd", "-fp", "--float-param=3.14", "--int-param", "3", 
                            "--multiple-str", "asd", "asdasd", "--flag-param", "1"])
    
    assert res["pos_1"] == "woot"
    assert res["pos_2"] == []
    assert res["pos_3"] == ["odn", "fpo", "sdnf"]
    assert res["pos_4"] == ["asd"]
    assert res["pure_flag"] is None
    assert res["string_param"] == "hello"
    assert res["int_param"] == 3
    assert res["multiple_str"] == ["asd", "asdasd"]
    assert res["flag_param"] == "1"

    with pytest.raises(Exception):
        res = params.from_cli(["woot", "odn", "fpo", "sdnf", "asd", "-l"])

    with pytest.raises(Exception):
        res = params.from_cli(["woot", "odn", "fpo", "sdnf", "asd", "-fpl"])

    with pytest.raises(ValueError):
        res = params.from_cli(["woot", "odn", "fpo", "sdnf", "asd", "--float-param", "asd"])

    with pytest.raises(Exception):
        res = params.from_cli(["woot", "odn"])

    with pytest.raises(Exception):
        res = params.from_cli(["woot", "odn", "fpo", "sdnf", "asd", "-fp", "asd"])

    # print()
    # params.print_cli_help()

def test_flag_param():
    params = parametric.ParameterManager()
    params.add_param(
        name="flag_param",
        help="A bool parameter used for testing.",
        behaviour=parametric.BooleanFlagBehaviour()
    )

    res = params.from_cli([])
    assert res["flag_param"] == False

    res = params.from_cli(["--flag-param"])
    assert res["flag_param"] == True

    res = params.from_cli(["--no-flag-param"])
    assert res["flag_param"] == False

    res = params.from_cli(["--flag-param", "--no-flag-param"])
    assert res["flag_param"] == False

    res = params.from_cli(["--no-flag-param", "--flag-param"])
    assert res["flag_param"] == True

def test_flags():
    with pytest.raises(ValueError):
        params = parametric.ParameterManager()
        params.add_param(
            name="string_param",
            flags=["-1"],
        )

    with pytest.raises(ValueError):
        params = parametric.ParameterManager()
        params.add_param(
            name="string_param",
            flags=["-as"],
        )

    with pytest.raises(ValueError):
        params = parametric.ParameterManager()
        params.add_param(
            name="string_param"
        )

        params.add_param(
            name="string_param"
        )

    with pytest.raises(ValueError):
        params = parametric.ParameterManager()
        params.add_param(
            name="string_param",
            flags="-a"
        )

        params.add_param(
            name="string_param2",
            flags="-a"
        )

    with pytest.raises(ValueError):
        params = parametric.ParameterManager()
        params.add_param(
            name="string_param",
            nargs="k"
        )

def test_parse_shared():
    params = parametric.ParameterManager()
    
    params.add_param(
        name="pos_1",
        help="A positional arg",
        cli_positional=True
    )

    params.add_param(
        name="string_param",
        flags=["--string-param"],
        help="A string parameter used for testing.",
    )

    params.add_param(
        name="multiple_str",
        flags=["--multiple-str"],
        help="A string parameter that can be added multiple times.",
        nargs="*",
        default=["hej"]
    )

    res, rest = params.from_shared_cli(["--string-param", "hello", "pos_value", "--string-param", "next", "pos_val_2"])

    assert res["string_param"] == "hello"
    assert res["pos_1"] == "pos_value"
    assert rest == ["--string-param", "next", "pos_val_2"]

    res, rest = params.from_shared_cli(rest)

    assert len(rest) == 0
    assert res["string_param"] == "next"
    assert res["pos_1"] == "pos_val_2"


    res, rest = params.from_shared_cli(["--multiple-str", "one", "two", "--", "pos_value", "--string-param", "next", "pos_val_2"])

    assert rest == ["--string-param", "next", "pos_val_2"]
    assert res["multiple_str"] == ["one", "two"]
    assert res["pos_1"] == "pos_value"
