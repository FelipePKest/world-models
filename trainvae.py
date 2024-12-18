""" Treinamento do VAE """
import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.vae import VAE

import matplotlib.pyplot as plt

from utils.misc import save_checkpoint, LSIZE, RED_SIZE
## ATENÇÃO: ISSO DEVE SER SUBSTITUÍDO PELO PYTORCH 0.5
from utils.learning import EarlyStopping, ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

# Analisador de argumentos da linha de comando
parser = argparse.ArgumentParser(description='Treinador do VAE')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Tamanho do lote de entrada para treinamento (padrão: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Número de épocas para treinar (padrão: 1000)')
parser.add_argument('--logdir', type=str, help='Diretório onde os resultados serão armazenados')
parser.add_argument('--noreload', action='store_true',
                    help='Não recarrega o melhor modelo, se especificado')
parser.add_argument('--nosamples', action='store_true',
                    help='Não salva amostras durante o treinamento, se especificado')

args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
# Corrige divergência numérica devido a bug no Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

# Transformações para os dados de treino e teste
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

# Carrega datasets de treino e teste
dataset_train = RolloutObservationDataset('datasets/carracing', transform_train, train=True)
dataset_test = RolloutObservationDataset('datasets/carracing', transform_test, train=False)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

# Inicializa o modelo, otimizador e outros componentes de treinamento
model = VAE(3, LSIZE).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

# Função de perda: Reconstrução + divergência KL
def loss_function(recon_x, x, mu, logsigma):
    """ Função de perda do VAE """
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD

# Função de treinamento por época
def train(epoch):
    """ Uma época de treinamento """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Treino Época: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tPerda: {loss.item() / len(data):.6f}')

    epoch_loss = train_loss / len(train_loader.dataset)
    print(f'====> Época: {epoch} Perda média: {epoch_loss:.4f}')
    return epoch_loss

# Função de teste
def test():
    """ Uma época de teste """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Perda no conjunto de teste: {test_loss:.4f}')
    return test_loss

# Criação dos diretórios do VAE se não existirem
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print(f"Recarregando modelo na época {state['epoch']}, com erro de teste {state['precision']:.4f}")
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])

cur_best = None
epoch_losses = []

for epoch in range(1, args.epochs + 1):
    epoch_losses.append(train(epoch))
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # Salvar checkpoint
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)

    # Salvar amostras
    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE), join(vae_dir, f'samples/sample_{epoch}.png'))

    if earlystopping.stop:
        print(f"Fim do treinamento por early stopping na época {epoch}")
        break

# Salvar perdas por época
with open(f"{vae_dir}_epoch_losses.txt", "w") as txt_file:
    for loss in epoch_losses:
        txt_file.write(f"{loss}\n")
