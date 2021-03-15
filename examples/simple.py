import pystematic

@pystematic.experiment
def my_experiment(parameters, context):
    print("Hello from my_experiment")

if __name__ == "__main__":
    pystematic.global_entrypoint()
