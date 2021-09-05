import time

import pytest
import pystematic
import pystematic.core
from pystematic import output_dir

import logging


def _list_contains_param_with_name(param_list, *param_names):
    existing_param_names = {param.name for param in param_list}

    return len(param_names) == len(existing_param_names.intersection(param_names))


def test_define_experiment():

    @pystematic.experiment
    def exp(params):
        pass

    assert isinstance(exp, pystematic.core.Experiment)
    assert exp.name == "exp"

    @pystematic.experiment(
        name="override"
    )
    def exp(params):
        pass

    assert exp.name == "override"


def test_main_function_is_run():

    class CustomException(Exception):
        pass

    @pystematic.experiment
    def exp(params):
        raise CustomException()

    with pytest.raises(CustomException):
        exp.cli([])

    with pytest.raises(CustomException):
        exp.run({})


def test_params_are_added():

    class CustomException(Exception):
        pass
    
    @pystematic.parameter(
        name="test_param"
    )
    @pystematic.experiment
    @pystematic.parameter(
        name="int_param",
        type=int
    )
    def exp(params):
        assert "test_param" in params
        assert params["test_param"] == "test"

        assert "int_param" in params
        assert params["int_param"] == 3
        raise CustomException()

    params = exp.get_parameters()
    
    assert _list_contains_param_with_name(params, "test_param", "int_param")
    
    with pytest.raises(CustomException):
        exp.cli(["--test-param", "test", "--int-param", "3"])

    with pytest.raises(CustomException):
        exp.run({"test_param": "test", "int_param": 3})


def test_duplicate_parameter_definition_raises_exception():
    with pytest.raises(Exception):
        @pystematic.parameter(
            name="param1"
        )
        @pystematic.experiment
        @pystematic.parameter(
            name="param1"
        )
        def exp(params):
            pass


def test_experiment_group():

    class Exp1Ran(Exception):
        pass

    class Exp2Ran(Exception):
        pass
    
    @pystematic.group
    def group(params):
        pass

    @pystematic.parameter(
        name="param1"
    )
    @group.experiment
    def exp1(params):
        assert params["param1"] == "value"
        raise Exp1Ran()

    @pystematic.parameter(
        name="param2"
    )
    @group.experiment
    def exp2(params):
        assert params["param2"] == "value"
        raise Exp2Ran()

    with pytest.raises(pystematic.core.Error):
        group.cli([], exit_on_error=False)

    with pytest.raises(Exp1Ran):
        group.cli(["exp1", "--param1", "value"])

    with pytest.raises(Exp2Ran):
        group.cli(["exp2", "--param2", "value"])


def test_output_dir_works():

    class CustomException(Exception):
        pass

    @pystematic.experiment
    def output_exp(params):
        with output_dir.joinpath("testfile.txt").open("w") as f:
            f.write("hello")

        raise CustomException()

    with pytest.raises(CustomException):
        output_exp.run({})


def test_param_matrix():
    param_list = pystematic.param_matrix(
        int_param=[1, 2],
        str_param=["hello", "world"]
    )
    assert param_list == [
        {
            "int_param": 1,
            "str_param": "hello"
        },
        {
            "int_param": 1,
            "str_param": "world"
        },
        {
            "int_param": 2,
            "str_param": "hello"
        },
        {
            "int_param": 2,
            "str_param": "world"
        }
    ]

    param_list = pystematic.param_matrix(
        int_param=[1, 2],
        str_param="hello"
    )
    assert param_list == [
        {
            "int_param": 1,
            "str_param": "hello"
        },
        {
            "int_param": 2,
            "str_param": "hello"
        }
    ]


def test_group_nesting():
    class ExpRan(Exception):
        pass

    @pystematic.group
    def group1(params):
        pass

    @group1.group
    def group2(params):
        pass

    @pystematic.parameter(
        name="param1"
    )
    @group2.experiment
    def exp(params):
        assert params["param1"] == "value"
        raise ExpRan()
    
    with pytest.raises(ExpRan):
        group1.cli(["group2", "exp", "--param1", "value"])


def test_cli_exit():
    class CustomException(Exception):
        pass
    
    @pystematic.parameter(
        name="test_param",
        required=True
    )
    @pystematic.parameter(
        name="int_param",
        type=int
    )
    @pystematic.experiment
    def exp(params):
        assert "test_param" in params
        assert params["test_param"] == "test"

        assert "int_param" in params
        assert params["int_param"] == 3
        raise CustomException()
    
    with pytest.raises(Exception):
        exp.cli(["--int-param", "3"], exit_on_error=False)


def test_experiment_nesting_raises_exception():


    @pystematic.parameter(
        name="str_param"
    )    
    @pystematic.experiment
    def exp1(params):
        exp2.run({
            "str_param": "exp2"
        })

    @pystematic.parameter(
        name="str_param"
    )
    @pystematic.experiment
    def exp2(params):
        pass

    with pytest.raises(pystematic.core.ExperimentError):
        exp1.run({
            "str_param": "exp1"
        })


def test_parameter_inheritence_error_handling():

    @pystematic.parameter(
        name="param1"
    )
    @pystematic.group
    def group(params):
        pass

    @pystematic.parameter(
        name="param1"
    )
    @pystematic.experiment
    def exp(params):
        pass
    
    with pytest.raises(Exception):
        @pystematic.parameter(
            name="param1"
        )
        @group.experiment
        def exp2(params):
            pass
    
    with pytest.raises(Exception):
        @pystematic.parameter(
            name="param1"
        )
        @group.experiment
        @pystematic.parameter(
            name="param1"
        )
        def exp2(params):
            pass

    with pytest.raises(Exception):
        @pystematic.parameter(
            name="param1"
        )
        @group.group
        def group2(params):
            pass
    
    with pytest.raises(Exception):
        @group.group
        @pystematic.parameter(
            name="param1"
        )
        def group2(params):
            pass

    with pytest.raises(Exception):
        @pystematic.parameter(
            name="param1"
        )
        @pystematic.experiment(
            inherit_params=exp
        )
        def exp2(params):
            pass

    with pytest.raises(Exception):
        @pystematic.experiment(
            inherit_params=exp
        )
        @pystematic.parameter(
            name="param1"
        )
        def exp2(params):
            pass


def test_experiment_inherit_params_from_group():

    class ExpRan(Exception):
        pass

    @pystematic.parameter(
        name="param1"
    )
    @pystematic.group
    def group1(params):
        pass

    @pystematic.parameter(
        name="param2"
    )
    @group1.group
    def group2(params):
        pass

    @pystematic.parameter(
        name="param3"
    )
    @group2.experiment
    def exp(params):
        assert params["param1"] == "value1"
        assert params["param2"] == "value2"
        assert params["param3"] == "value3"
        raise ExpRan()

    assert _list_contains_param_with_name(exp.get_parameters(), "param1", "param2", "param3")
    
    with pytest.raises(ExpRan):
        group1.cli(["group2", "exp", "--param1", "value1", "--param2", "value2", "--param3", "value3"])


def test_param_groups():

    class ExpRan(Exception):
        pass

    @pystematic.param_group("agroup",
        pystematic.parameter(
            name="param2"
        ),
        pystematic.parameter(
            name="param3"
        )
    )
    @pystematic.parameter(
        name="param1"
    )
    @pystematic.experiment
    def exp(params):
        assert params["param1"] == "value1"
        assert params["param2"] == "value2"
        assert params["param3"] == "value3"
        raise ExpRan()

    with pytest.raises(ExpRan):
        exp.run({
            "param1": "value1",
            "param2": "value2",
            "param3": "value3",
        })

    with pytest.raises(ExpRan):
        exp.cli(["--param1", "value1", "--param2", "value2", "--param3", "value3"])


def test_param_group_inheritence():
    class ExpRan(Exception):
        pass

    @pystematic.parameter(
        name="param1"
    )
    @pystematic.param_group(
        "param_group1",
        pystematic.parameter(
            name="param11"
        ),
        pystematic.parameter(
            name="param12"
        )
    )
    @pystematic.group
    def group1(params):
        pass

    @pystematic.param_group(
        "param_group2",
        pystematic.parameter(
            name="param21"
        ),
        pystematic.parameter(
            name="param22"
        )
    )
    @pystematic.parameter(
        name="param2"
    )
    @group1.group
    def group2(params):
        pass

    @pystematic.param_group(
        "param_group3",
        pystematic.parameter(
            name="param31"
        ),
        pystematic.parameter(
            name="param32"
        )
    )
    @pystematic.parameter(
        name="param3"
    )
    @group2.experiment
    def exp(params):
        assert params["param1"] == "value1"
        assert params["param2"] == "value2"
        assert params["param3"] == "value3"

        assert params["param11"] == "value1"
        assert params["param12"] == "value2"

        assert params["param21"] == "value1"
        assert params["param22"] == "value2"

        assert params["param31"] == "value1"
        assert params["param32"] == "value2"


        raise ExpRan()

    assert _list_contains_param_with_name(
            exp.get_parameters(), 
            "param1", "param2", "param3", "param11", "param12", 
            "param21", "param22", "param31", "param32"
        )
    
    with pytest.raises(ExpRan):
        group1.cli(["group2", "exp", 
                    "--param1", "value1", "--param2", "value2", "--param3", "value3",
                    "--param11", "value1", "--param12", "value2",
                    "--param21", "value1", "--param22", "value2",
                    "--param31", "value1", "--param32", "value2"])


    
    @pystematic.param_group("param_group1",
        pystematic.parameter(
            name="param1"
        ),
    )
    @pystematic.group
    def group1(params):
        pass
    
    with pytest.raises(pystematic.core.Error):
        @pystematic.parameter(
            name="param1"
        )
        @group1.experiment
        def exp(params):
            pass

    with pytest.raises(pystematic.core.Error):
        @group1.experiment
        @pystematic.parameter(
            name="param1"
        )
        def exp(params):
            pass


def test_multiple_inheritence():

    @pystematic.parameter(
        name="param_exp1"
    )
    def exp1(params):
        pass

    @pystematic.param_group(
        pystematic.parameter(
            name="param_exp2"
        )
    )
    def exp2(params):
        pass


    @pystematic.experiment(inherit_params=[exp1, exp2])
    def exp3(params):
        pass


def test_group_multiple_inheritence():
    @pystematic.param_group("group_a",
        pystematic.parameter(
            name="param1"
        ),
    )
    @pystematic.experiment
    def exp1(params):
        pass

    @pystematic.param_group("group_b",
        pystematic.parameter(
            name="param1"
        ),
    )
    @pystematic.experiment
    def exp2(params):
        pass


    with pytest.raises(pystematic.core.Error):
        @pystematic.experiment(inherit_params=[exp1, exp2])
        def exp3(params):
            pass


def test_param_group_duplicates():
    with pytest.raises(pystematic.core.Error):
        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        @pystematic.parameter(
            name="param1"
        )
        @pystematic.experiment
        def exp(params):
            pass

    with pytest.raises(pystematic.core.Error):
        @pystematic.parameter(
            name="param1"
        )
        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        @pystematic.experiment
        def exp(params):
            pass

    with pytest.raises(pystematic.core.Error):

        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        @pystematic.experiment
        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        def exp(params):
            pass

    with pytest.raises(pystematic.core.Error):

        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        @pystematic.experiment
        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param2"
            ),
        )
        def exp(params):
            pass

    with pytest.raises(pystematic.core.Error):

        @pystematic.param_group("agroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        @pystematic.experiment
        @pystematic.param_group("bgroup",
            pystematic.parameter(
                name="param1"
            ),
        )
        def exp(params):
            pass


def test_launch_subprocess():
    _subprocess_exp.run({})


@pystematic.experiment
def _subprocess_exp(params):
    logger = logging.getLogger("subprocess_exp")
    if not pystematic.is_subprocess():
        procs = [pystematic.launch_subprocess() for _ in range(3)]

    logger.info(pystematic.local_rank())
    
    if not pystematic.is_subprocess():
        for proc in procs:
            proc.join()


def test_param_sweep():
    pystematic.run_parameter_sweep(_sweep_exp, [{}]*5, max_num_processes=2)


@pystematic.experiment
def _sweep_exp(params):
    logger = logging.getLogger("test_param_sweep")
    logger.info("NEW")
    time.sleep(0.1)
    logger.info("END")


def test_load_param_file():

    class ExpRan(Exception):
        pass

    @pystematic.parameter(
        name="param1",
        type=str
    )
    @pystematic.parameter(
        name="param2",
        type=int
    )
    @pystematic.experiment
    def exp1(params):
        assert params["param1"] == "hello"
        assert params["param2"] == 10
        assert str(params["params_file"]) == "tests/resources/params1.yaml"
        raise ExpRan

    with pytest.raises(ExpRan):
        exp1.run({
            "params_file": "tests/resources/params1.yaml"
        })

    @pystematic.parameter(
        name="param1",
        type=str
    )
    @pystematic.parameter(
        name="param2",
        type=int
    )
    @pystematic.experiment
    def exp2(params):
        assert params["param1"] == "world"
        assert params["param2"] == 20
        assert str(params["params_file"]) == "tests/resources/params2.yaml"
        raise ExpRan

    with pytest.raises(ExpRan):
        exp2.run({
            "params_file": "tests/resources/params2.yaml"
        })
