import argparse
import sys
import os
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ThreadPoolExecutor
import subprocess
from datetime import datetime
import time

from experiments import all_experiments

root_dir = "{}/..".format(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Experiment Launcher')
parser.add_argument('--experiment', type=str, default="dann_imput_usps_mnist")
parser.add_argument('--debug', type=str, default="false")
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()


def launcher_with_environment(env, debug):
    def launch_command_line(command):
        tab = command.split()
        print("Executing {}".format(command))
        if not debug:
            print(tab)
            try:
                myPopen = subprocess.Popen(
                    tab,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                for l in myPopen.stderr:
                    print(l)
            except subprocess.CalledProcessError as e:
                print(e.output)
    return launch_command_line


# Retrieving experiment
experiment_name = args.experiment
experiment = all_experiments[experiment_name]
gpu_id = args.gpu_id

# Creating env for the runs
env = os.environ.copy()
print("Using env {}".format(env))

# Checking how many tests we want to launch
all_configs = list(ParameterGrid(experiment.GRID))


nb_tests = len(all_configs)
print("%d experiments to launch..." % nb_tests)

# Creating executors with max nb processes from the config
executor = ThreadPoolExecutor(max_workers=experiment.MAX_NB_PROCESSES)

# Running the tests
with open("params_to_remember.txt", "a") as f_params:
    f_params.write("------------\n")

    now = datetime.now()
    ts = int(time.mktime(now.timetuple()))

    f_params.write("Starting run at {} (metarunid {})\n".format(str(now), str(ts)))
    for runid, parameter_set in enumerate(all_configs):
        print(parameter_set)
        f_params.write("{} => {}\n".format(runid, parameter_set))
        f_params.flush()

        # The python binary is available in sys.executable
        args = ["{} {}".format(sys.executable, "{}/{}".format(root_dir, experiment.BINARY))]
        for a in parameter_set:
            args.append("-" + a + " " + str(parameter_set[a]))
        args.append("--run_id {}".format(str(runid)))
        # The experiment should be aware of the number of running processes so that it does not
        # ask for too much memory on the GPU
        args.append("--max_nb_processes {}".format(min([experiment.MAX_NB_PROCESSES, nb_tests])))
        args.append("--experiment_name {}".format(experiment_name))
        args.append("--metarun_id {}".format(str(ts)))
        args.append("--gpu_id {}".format(gpu_id))

        command = " ".join(args)
        executor.submit(launcher_with_environment(env, experiment.DEBUG), command)

