"""
Define o modelo MDRNN, usado como o modulo de memoria no Modelo de Mundo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
    """
    Calcula a perda de Mistura de Modelos Gaussianos (GMM).

    Computa o log negativo da probabilidade do `batch` sob o modelo GMM
    descrito por `mus`, `sigmas` e `logpi`.

    Parâmetros:
    - batch (torch.Tensor): Tensor de dados com forma (bs1, bs2, ..., fs).
    - mus (torch.Tensor): Tensor de médias com forma (bs1, bs2, ..., gs, fs).
    - sigmas (torch.Tensor): Tensor de desvios padrão (bs1, bs2, ..., gs, fs).
    - logpi (torch.Tensor): Tensor dos logaritmos das probabilidades (bs1, ..., gs).
    - reduce (bool): Se `True`, retorna a média da perda.

    Retorno:
    - torch.Tensor: Perda GMM calculada.
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob)
    return - log_prob

class _MDRNNBase(nn.Module):
    """
    Classe base para MDRNN.

    Define atributos comuns como tamanho dos latentes, ações, estados ocultos e gaussianos.
    """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        """
        Método a ser implementado nas subclasses.
        """
        pass

class MDRNN(_MDRNNBase):
    """
    MDRNN para previsao de multiplos passos.

    Utiliza LSTM para processar sequência de latentes e ações.
    """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """Realiza a previsão para múltiplos passos.

        Parâmetros:
        - actions (torch.Tensor): Ações com forma (SEQ_LEN, BSIZE, ASIZE).
        - latents (torch.Tensor): Latentes com forma (SEQ_LEN, BSIZE, LSIZE).

        :returns:
        - mus, sigmas, logpi, rs, ds: parâmetros GMM, previsao da reward e previsão do terminal.

        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """  
    MDRNN para um único passo.

    Utiliza LSTMCell para previsão em uma única etapa. 
    """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """  Realiza a previsão para um único passo.

        Parâmetros:
        - action (torch.Tensor): Ação com forma (BSIZE, ASIZE).
        - latent (torch.Tensor): Latente com forma (BSIZE, LSIZE).
        - hidden (tuple): Estado oculto da LSTM com (BSIZE, RSIZE).

        :returns: - mus, sigmas, logpi, r, d, next_hidden: parâmetros GMM previsao da reward previsão do terminal próximos estados.
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
