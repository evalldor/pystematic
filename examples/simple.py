import pystematic as ps

@ps.experiment
def my_experiment(params):
    print("Hello from my_experiment")


if __name__ == "__main__":
    my_experiment.cli()
