import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')
parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain return estimate.')
parser.add_argument('--pop-size', type=int, help='Population size.')
parser.add_argument('--target-return', type=float, help='Stops once the return gets above target_return')
parser.add_argument('--display', action='store_true', help="Use progress bars if specified.")
args = parser.parse_args()

# Multiprocessing variables (modified for serial execution)
n_samples = args.n_samples
pop_size = args.pop_size
time_limit = 1000

# Create tmp dir if non existent and clean it if existent
tmp_dir = join(args.logdir, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# Create ctrl dir if non existent
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

################################################################################
#                           Evaluation                                         #
################################################################################
def evaluate(r_gen, best_guess, rollouts=100):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args r_gen: Rollout generator
    :args best_guess: Best parameters guess
    :args rollouts: Number of rollouts

    :returns: minus averaged cumulated reward
    """
    restimates = []
    for _ in range(rollouts):
        result = r_gen.rollout(best_guess)
        restimates.append(result)

    return np.mean(restimates), np.std(restimates)

################################################################################
#                           Launch CMA                                         #
################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
controller = Controller(LSIZE, RSIZE, ASIZE).to(device)
r_gen = RolloutGenerator(args.logdir, device, time_limit)

# Define current best and load parameters
cur_best = None
ctrl_file = join(ctrl_dir, 'best.tar')
print("Attempting to load previous best...")
if exists(ctrl_file):
    state = torch.load(ctrl_file, map_location='cpu')
    cur_best = - state['reward']
    controller.load_state_dict(state['state_dict'])
    print("Previous best was {}...".format(-cur_best))

parameters = controller.parameters()
es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {'popsize': pop_size})

epoch = 0
log_step = 3
while not es.stop():
    print("Running epoch {}".format(epoch))
    if cur_best is not None and - cur_best > args.target_return:
        print("Already better than target, breaking...")
        break

    r_list = [0] * pop_size  # Result list
    solutions = es.ask()

    # Directly evaluate solutions in serial
    for s_id, s in enumerate(solutions):
        print("Checking solution ", str(s_id))
        results = []
        for _ in range(n_samples):
            result = r_gen.rollout(s)
            results.append(result)
        r_list[s_id] = np.mean(results)

    es.tell(solutions, r_list)
    es.disp()

    # Evaluation and saving
    if epoch % log_step == log_step - 1:
        best_params = solutions[np.argmin(r_list)]
        best, std_best = evaluate(r_gen, best_params)
        print("Current evaluation: {}".format(best))
        if not cur_best or cur_best > best:
            cur_best = best
            print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
            load_parameters(best_params, controller)
            torch.save(
                {'epoch': epoch, 'reward': - cur_best, 'state_dict': controller.state_dict()},
                join(ctrl_dir, 'best.tar')
            )
        if -best > args.target_return:
            print("Terminating controller training with value {}...".format(best))
            break

    epoch += 1

es.result_pretty()
