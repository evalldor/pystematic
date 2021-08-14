import pytest
import pystematic
from pystematic import output_dir

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