import pystematic as ps

@ps.parameter(
    name="group_param"
)
@ps.group
def my_group():
    pass

@my_group.group
def my_group2():
    pass

@my_group.experiment
def my_experiment(params):
    print("Hello from my_experiment")

@my_group2.experiment
def my_experiment2(params):
    print("Hello from my_experiment2")

@my_group2.experiment
def my_experiment3(params):
    print("Hello from my_experiment3")

@my_group2.group
def my_group3():
    pass

@my_group3.experiment
def my_experiment4(params):
    print("Hello from my_experiment4")

@my_group3.experiment
def my_experiment5(params):
    print("Hello from my_experiment5")

@my_group3.experiment
def my_experiment6(params):
    print("Hello from my_experiment6")


if __name__ == "__main__":
    my_group.cli()
