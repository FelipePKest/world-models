""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """
    Classe Controller: Implementa a camada de Controller, do artigo World Models.

    Herda de:
        nn.Module (PyTorch)
    """
    def __init__(self, latents, recurrents, actions):
        """
        Inicializa a classe Controller.

        Parâmetros:
        - latents (int): Dimensao das variaveis latentes definidas pela camada de Visao.
        - recurrents (int): Dimensao das variáveis recorrentes definidas pela camada de Memoria.
        - actions (int): Número de ações (saídas) que o modelo deve executar.

        Atributos:
        - self.fc (nn.Linear): Camada totalmente conectada que concatena as variaveis latentes e recorrentes
        para indicar a acao.
        """
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        """
        Define a passagem direta da rede neural.

        Parâmetros:
        - *inputs (tuple de torch.Tensor): Entradas que incluem tensores de latentes
          e recorrentes.

        Retorno:
        - torch.Tensor: Saída da camada linear após concatenar as entradas e passar pela camada totalmente conectada.
        """
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)
