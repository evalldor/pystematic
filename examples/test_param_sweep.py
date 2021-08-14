import time
import random
import pystematic as ps
import pickle


@ps.experiment
def main_experiment(params):
    print("NEW")
    time.sleep(random.uniform(0, 10))
    print("END")


@ps.experiment
def param_search(params):
    
    ps.run_parameter_sweep(main_experiment, [{}]*50, max_num_processes=10)


if __name__ == "__main__":
    param_search.cli()
