import pytest

import pystematic.core as core

def test_main_function_is_run():

    class CustomException(Exception):
        pass

    def main_function(params):
        raise CustomException()

    exp = core.Experiment(main_function)
    
    with pytest.raises(CustomException):
        exp.cli([])

    with pytest.raises(CustomException):
        exp.run({})


def test_params_are_added():

    class CustomException(Exception):
        pass
    
    @core.parameter_decorator(
        name="test_param"
    )
    @core.parameter_decorator(
        name="int_param",
        type=int
    )
    def main_function(params):
        assert "test_param" in params
        assert params["test_param"] == "test"

        assert "int_param" in params
        assert params["int_param"] == 3
        raise CustomException()

    exp = core.Experiment(main_function)
    
    with pytest.raises(CustomException):
        exp.cli(["--test-param", "test", "--int-param", "3"])

    with pytest.raises(CustomException):
        exp.run({"test_param": "test", "int_param": 3})


def test_experiment_group():

    class Exp1Ran(Exception):
        pass

    class Exp2Ran(Exception):
        pass
    
    @core.parameter_decorator(
        name="param1"
    )
    def exp1(params):
        assert params["param1"] == "value"
        raise Exp1Ran()

    @core.parameter_decorator(
        name="param2"
    )
    def exp2(params):
        assert params["param2"] == "value"
        raise Exp2Ran()

    exp_one = core.Experiment(exp1)
    exp_two = core.Experiment(exp2)

    group = core.ExperimentGroup(lambda x:x)
    group.add_experiment(exp_one)
    group.add_experiment(exp_two)

    with pytest.raises(Exp1Ran):
        group.cli(["exp1", "--param1", "value"])

    with pytest.raises(Exp2Ran):
        group.cli(["exp2", "--param2", "value"])
