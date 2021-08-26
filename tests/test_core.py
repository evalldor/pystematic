import pytest
import pystematic
import pystematic.core
from pystematic import output_dir



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

    def _list_contains_param_with_name(param_list, param_name):

        for param in param_list:
            if param.name == param_name:
                return True

        return False

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
    
    assert _list_contains_param_with_name(params, "test_param")
    assert _list_contains_param_with_name(params, "int_param")
    
    with pytest.raises(CustomException):
        exp.cli(["--test-param", "test", "--int-param", "3"])

    with pytest.raises(CustomException):
        exp.run({"test_param": "test", "int_param": 3})


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

    with pytest.raises(Exception):
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

def test_experiment_nesting():


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

def test_experiment_inherit_params_from_group():

    def _list_contains_param_with_name(param_list, param_name):

        for param in param_list:
            if param.name == param_name:
                return True

        return False

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

    assert _list_contains_param_with_name(exp.get_parameters(), "param1")
    assert _list_contains_param_with_name(exp.get_parameters(), "param2")
    assert _list_contains_param_with_name(exp.get_parameters(), "param3")
    
    with pytest.raises(ExpRan):
        group1.cli(["group2", "exp", "--param1", "value1", "--param2", "value2", "--param3", "value3"])
