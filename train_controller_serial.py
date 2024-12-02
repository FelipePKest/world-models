import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir
from time import sleep
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters, flatten_parameters

# Análise dos argumentos de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Local onde tudo será armazenado.')
parser.add_argument('--n-samples', type=int, help='Número de amostras usadas para estimar o retorno.')
parser.add_argument('--pop-size', type=int, help='Tamanho da população.')
parser.add_argument('--target-return', type=float, help='Interrompe quando o retorno ultrapassa o alvo.')
parser.add_argument('--display', action='store_true', help="Exibe barras de progresso, se especificado.")
args = parser.parse_args()

# Variáveis de multiprocessamento (modificado para execução serial)
n_samples = args.n_samples
pop_size = args.pop_size
time_limit = 1000

# Cria o diretório tmp se não existir e limpa se já existir
tmp_dir = join(args.logdir, 'tmp')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

# Cria o diretório ctrl se não existir
ctrl_dir = join(args.logdir, 'ctrl')
if not exists(ctrl_dir):
    mkdir(ctrl_dir)

################################################################################
#                           Avaliação                                          #
################################################################################
def evaluate(r_gen, best_guess, rollouts=100):
    """ Avalia o controlador atual.

    A avaliação retorna o negativo da recompensa acumulada média ao longo dos testes.

    :args r_gen: Gerador de rollouts
    :args best_guess: Melhor estimativa de parâmetros
    :args rollouts: Número de rollouts

    :returns: negativo da média da recompensa acumulada
    """
    restimates = []
    for _ in range(rollouts):
        result = r_gen.rollout(best_guess)
        restimates.append(result)

    return np.mean(restimates), np.std(restimates)

################################################################################
#                           Iniciar CMA                                        #
################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
controller = Controller(LSIZE, RSIZE, ASIZE).to(device)
r_gen = RolloutGenerator(args.logdir, device, time_limit)

# Define o melhor atual e carrega os parâmetros
cur_best = None
ctrl_file = join(ctrl_dir, 'best.tar')
print("Tentando carregar o melhor anterior...")
if exists(ctrl_file):
    state = torch.load(ctrl_file, map_location='cpu')
    cur_best = -state['reward']
    controller.load_state_dict(state['state_dict'])
    print(f"Melhor anterior era {cur_best}...")

parameters = controller.parameters()
es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {'popsize': pop_size})

epoch = 0
log_step = 3
while not es.stop():
    print("Rodando epoch {}".format(epoch))
    if cur_best is not None and - cur_best > args.target_return:
        print("Melhor que o alvo, interrompendo...")
        break

    r_list = [0] * pop_size  # Lista de resultados
    solutions = es.ask()

    # Avaliação direta das soluções em série
    for s_id, s in enumerate(solutions):
        print("Verificando solução", s_id)
        results = []
        for _ in range(n_samples):
            result = r_gen.rollout(s)
            results.append(result)
        r_list[s_id] = np.mean(results)

    es.tell(solutions, r_list)
    es.disp()

    # Avaliação e salvamento
    if epoch % log_step == log_step - 1:
        best_params = solutions[np.argmin(r_list)]
        best, std_best = evaluate(r_gen, best_params)
        print(f"Avaliação atual: {best}")
        if not cur_best or cur_best > best:
            cur_best = best
            print(f"Salvando novo melhor com valor {cur_best}±{std_best}...")
            load_parameters(best_params, controller)
            torch.save(
                {'epoch': epoch, 'reward': -cur_best, 'state_dict': controller.state_dict()},
                join(ctrl_dir, 'best.tar')
            )
        if -best > args.target_return:
            print(f"Encerrando treinamento com valor {best}...")
            break

    epoch += 1

es.result_pretty()
