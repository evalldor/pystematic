import time
import random
import pystematic



@pystematic.experiment
def main_experiment(params, ctx):
    print("NEW")
    time.sleep(random.uniform(0, 10))
    print("END")


@pystematic.experiment
def param_search(params, ctx):
    
    pystematic.run_parameter_sweep(main_experiment, [{}]*50, max_num_processes=10)

if __name__ == "__main__":
    pystematic.global_entrypoint()
