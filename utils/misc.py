""" 
Várias utilidades auxiliares para suportar o treinamento e avaliação do modelo do mundo. 
Inclui amostragem de políticas, salvamento/carregamento de checkpoints, manipulação de parâmetros 
e geração de rollouts usando VAE, MDRNN e um controlador no ambiente CarRacing. 
"""
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np

from utils.controller import Controller
from utils.vae import VAE
from utils.mdrnn import MDRNNCell

import gymnasium as gym
import gymnasium.envs.box2d as box2d

# Modifica o tamanho da representação de estado do ambiente CarRacing
box2d.car_racing.STATE_W, box2d.car_racing.STATE_H = 64, 64

# Constantes que definem os tamanhos das ações, latentes, estados ocultos e imagens reduzidas
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

# Pipeline de transformação de imagem para observações do ambiente CarRacing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Roda uma etapa da politica simulada por um movimento browniano
    Args:
        action_space: Espaço de ação do Gym.
        seq_len: Número de ações na sequência.
        dt: Discretização temporal.
    Returns:
        Lista de ações amostradas da política.
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ 
    Salva o estado atual em um arquivo e atualiza o melhor checkpoint quando for o caso 
    Args:
        state: Estado a ser salvo.
        is_best: Indica se este é o melhor estado.
        filename: Caminho para salvar o estado atual.
        best_filename: Caminho para salvar o melhor estado.
    """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ 
    Achata os parâmetros do modelo em um único tensor 1D. 
    Args:
        params: Gerador de parâmetros do modelo.
    Returns:
        Array 1D numpy com os parâmetros achatados.
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ 
    Restaura os parâmetros aos formatos originais. 
    Args:
        params: Array 1D numpy de parâmetros achatados.
        example: Gerador de parâmetros com os formatos originais.
        device: Dispositivo onde os parâmetros serão colocados.
    Returns:
        Lista de tensores com os parâmetros no formato original.
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ 
    Carrega parâmetros achatados no modulo Controller. 
    Args:
        params: Array 1D numpy de parâmetros.
        controller: Modelo onde os parâmetros serão carregados.
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ 
    Gera os rollouts como exemplos usando os modulos de Visao e Memoria previamente treinados

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device="cpu", time_limit=1000, render_mode="rgb_array"):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v2', render_mode=render_mode)
        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ 
        Codifica a observação e computa a ação e próximo estado oculto. 
        Args:
            obs: Tensor da observação atual.
            hidden: Tensor do estado oculto atual.
        Returns:
            Tupla contendo a ação e o próximo estado oculto.
        """
        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().detach().numpy(), next_hidden

    def rollout(self, params, render=False):
        """ 
        Executa um rollout e retorna a recompensa cumulativa negativa. 
        Args:
            params: Array 1D numpy de parâmetros do controlador.
            render: Booleano para ativar a renderização do ambiente.
        Returns:
            Recompensa cumulativa negativa.
        """

        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()
        obs = transform(obs[0]).unsqueeze(0).to(self.device)

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, a, b = self.env.step(action)
            obs = transform(obs).unsqueeze(0).to(self.device)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return - cumulative
            i += 1

