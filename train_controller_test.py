import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from time import sleep
import torch
from torchvision import transforms
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters
import os

ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])
# parsing
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')
parser.add_argument('--n-samples', type=int, help='Number of samples used to obtain '
                    'return estimate.')
parser.add_argument('--pop-size', type=int, help='Population size.')
parser.add_argument('--target-return', type=float, help='Stops once the return '
                    'gets above target_return')
parser.add_argument('--display', action='store_true', help="Use progress bars if "
                    "specified.")
parser.add_argument('--max-workers', type=int, help='Maximum number of workers.',
                    default=32)
args = parser.parse_args()

# multiprocessing variables
n_samples = args.n_samples
pop_size = args.pop_size
num_workers = min(args.max_workers, n_samples * pop_size)
time_limit = 1000

# create tmp dir if non existent and clean it if existent
tmp_dir = join(args.logdir, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# create ctrl dir if non exitent

print("Creating directories")

ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)


################################################################################
#                           Serial routine                                     #
################################################################################
def serial_routine(params, time_limit):
    """ Serial routine.

    Executes the corresponding rollout for the given parameters.

    :args params: parameters to evaluate
    :args device: the device to use for computation
    :args time_limit: time limit for the rollout

    :returns: the result of the rollout
    """
    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, time_limit=time_limit)
        result = r_gen.rollout(params)
    return result


################################################################################
#                Evaluation                                                    #
################################################################################
def evaluate(solutions, results, rollouts=100):
    """ Give current controller evaluation.

    Evaluation is minus the cumulated reward averaged over rollout runs.

    :args solutions: CMA set of solutions
    :args results: corresponding results
    :args rollouts: number of rollouts

    :returns: minus averaged cumulated reward
    """
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    print("Evaluating...")
    for _ in tqdm(range(rollouts)):
        restimates.append(serial_routine(best_guess, time_limit))

    return best_guess, np.mean(restimates), np.std(restimates)

################################################################################
#                           Launch CMA                                         #
################################################################################
controller = Controller(LSIZE, RSIZE, ASIZE)  # dummy instance

# define current best and load parameters
cur_best = None
ctrl_file = join(ctrl_dir, 'best.tar')
print("Attempting to load previous best...")
if exists(ctrl_file):
    print("File exists")
    state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
    cur_best = - state['reward']
    controller.load_state_dict(state['state_dict'])
    print("Previous best was {}...".format(-cur_best))
else: 
    print("Start from scratch")
parameters = controller.parameters()

es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {'popsize': pop_size})
print("Start evolution...")

r_gen = RolloutGenerator(args.logdir, time_limit=100)
generations = 50000
log_step = 3
for gen in range(generations):
    print(f"Generation {gen}")
    solutions = es.ask()
    results = []
    print("Starting serial evaluation...")
    for s_id, solution in enumerate(solutions):
        # result = serial_routine(solution, time_limit)
        rewards = []
        with torch.no_grad():

            obs = r_gen.env.reset()
            obs = transform(obs[0]).unsqueeze(0).to(r_gen.device)
        
            # This first render is required !
            r_gen.env.render()

            hidden = [
                torch.zeros(1, RSIZE).to(r_gen.device)
                for _ in range(2)]

            cumulative = 0
            gen = 0

            while True:
                action, hidden = r_gen.get_action_and_transition(obs, hidden)
                obs, reward, done, a, b = r_gen.env.step(action)
                obs = transform(obs).unsqueeze(0).to(r_gen.device)

                rewards.append(reward)
                cumulative += reward
                if done or gen > r_gen.time_limit:
                    # return - cumulative
                    results.append(-cumulative)
                    break

                gen+= 1
                # best_guess, mean_reward, std_reward = evaluate(solutions, results)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    print(f" Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    print("Res len",results)
    if gen % log_step == log_step - 1:
        # best_params, best, std_best = evaluate(solutions, mean_rewards)
        print("Current evaluation: {}".format(mean_reward))
        if not cur_best or cur_best > mean_reward:
            cur_best = mean_reward
            print("Saving new best with value {}+-{}...".format(-cur_best, std_reward))
            load_parameters(best_guess, controller)
            torch.save(
                {'epoch': gen,
                 'reward': - cur_best,
                 'state_dict': controller.state_dict()},
                join(ctrl_dir, 'best.tar'))
        if - cur_best > args.target_return:
            print("Terminating controller training with value {}...".format(cur_best))
            break


    gen += 1
    es.tell(solutions, results)
    es.logger.add()
    es.disp()

    if es.stop():
        print("stop")