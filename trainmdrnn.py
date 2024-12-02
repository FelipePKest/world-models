""" Treinamento de modelo recorrente """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping
## ATENÇÃO: ISSO DEVE SER SUBSTITUÍDO POR PYTORCH 0.5
from utils.learning import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("Treinamento MDRNN")
parser.add_argument('--logdir', type=str,
                    help="Onde os logs são armazenados e os modelos são carregados.")
parser.add_argument('--noreload', action='store_true',
                    help="Não recarregar se especificado.")
parser.add_argument('--include_reward', action='store_true',
                    help="Adicionar um termo de modelagem de recompensa à perda.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constantes
BSIZE = 16
SEQ_LEN = 32
epochs = 100

# Carregando VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "Nenhum VAE treinado no diretório de logs..."
state = torch.load(vae_file)
print(f"Carregando VAE na época {state['epoch']} com erro de teste {state['precision']}")

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Carregando modelo MDRNN
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print(f"Carregando MDRNN na época {rnn_state['epoch']} com erro de teste {rnn_state['precision']}")
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])

# Carregamento dos dados
transform = transforms.Lambda(lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8)

def to_latent(obs, next_obs):
    """ Transforma as observações para o espaço latente.

    :args obs: tensor 5D torch (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: tensor 5D torch (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: tensor 4D torch (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: tensor 4D torch (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool):
    """ Calcular as perdas.

    A perda que é calculada é:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(recompensa, recompensa_predita) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    O fator LSIZE + 2 é usado para compensar o fato de que o GMMLoss escala
    aproximadamente de forma linear com LSIZE. Todas as perdas são médias tanto nas dimensões
    do lote quanto da sequência (as duas primeiras dimensões).

    :args latent_obs: tensor torch (BSIZE, SEQ_LEN, LSIZE)
    :args acao: tensor torch (BSIZE, SEQ_LEN, ASIZE)
    :args recompensa: tensor torch (BSIZE, SEQ_LEN)
    :args latent_next_obs: tensor torch (BSIZE, SEQ_LEN, LSIZE)

    :returns: dicionário de perdas, contendo o gmm, o mse, o bce e
        a perda média.
    """
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epoch, train, include_reward): # pylint: disable=too-many-locals
    """ Uma passada pelos dados """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

epoch_losses = []
cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    epoch_losses.append(test_loss)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)

    if earlystopping.stop:
        print(f"Fim do treinamento devido ao early stopping na época {e}")
        break


with open(rnn_dir+"_epoch_losses.txt", "w") as txt_file:
    for loss in epoch_losses:
        txt_file.write(" ".join([str(loss),"\n"]))