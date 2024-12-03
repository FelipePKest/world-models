
"""
Variational autoencoder (VAE), usado como modelo visual 
para o nosso modelo do mundo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):  
    """Decodificador do VAE.  

    Reconstrói a imagem original a partir da codificação latente. Utiliza camadas totalmente conectadas e convoluções transpostas para expandir a representação latente até a dimensão original da imagem.  

    Attributes:  
        latent_size (int): Tamanho do vetor latente que serve como entrada.  
        img_channels (int): Número de canais da imagem de saída (ex.: 3 para RGB).  
        fc1 (nn.Linear): Camada totalmente conectada que mapeia o vetor latente para um vetor maior.  
        deconv1 (nn.ConvTranspose2d): Primeira camada de convolução transposta para reconstrução da imagem.  
        deconv2 (nn.ConvTranspose2d): Segunda camada de convolução transposta.  
        deconv3 (nn.ConvTranspose2d): Terceira camada de convolução transposta.  
        deconv4 (nn.ConvTranspose2d): Última camada de convolução transposta para saída final da imagem.  
    """  

    def __init__(self, img_channels, latent_size):  
        """Inicializa o Decoder com as camadas necessárias."""  
        super(Decoder, self).__init__()  
        self.latent_size = latent_size  
        self.img_channels = img_channels  

        self.fc1 = nn.Linear(latent_size, 1024)  
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)  
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)  
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)  
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)  

    def forward(self, x):  
        """Reconstrói a imagem a partir da representação latente.  

        Args:  
            x (Tensor): Vetor latente de entrada.  

        Returns:  
            Tensor: Imagem reconstruída com dimensões e canais originais.  
        """  
        x = F.relu(self.fc1(x))  
        x = x.unsqueeze(-1).unsqueeze(-1)  
        x = F.relu(self.deconv1(x))  
        x = F.relu(self.deconv2(x))  
        x = F.relu(self.deconv3(x))  
        reconstruction = F.sigmoid(self.deconv4(x))  
        return reconstruction  


class Encoder(nn.Module):  
    """Codificador do VAE.  

    Comprime uma imagem de entrada em uma representação latente compacta. Utiliza camadas convolucionais para extrair características relevantes.  

    Attributes:  
        latent_size (int): Tamanho do vetor latente gerado.  
        img_channels (int): Número de canais da imagem de entrada.  
        conv1 (nn.Conv2d): Primeira camada convolucional para extração de características.  
        conv2 (nn.Conv2d): Segunda camada convolucional.  
        conv3 (nn.Conv2d): Terceira camada convolucional.  
        conv4 (nn.Conv2d): Quarta camada convolucional.  
        fc_mu (nn.Linear): Camada totalmente conectada que gera a média da distribuição latente.  
        fc_logsigma (nn.Linear): Camada que gera o log da variância da distribuição latente.  
    """  

    def __init__(self, img_channels, latent_size):  
        """Inicializa o Encoder com camadas convolucionais e totalmente conectadas."""  
        super(Encoder, self).__init__()  
        self.latent_size = latent_size  
        self.img_channels = img_channels  

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)  
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)  
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)  

        self.fc_mu = nn.Linear(2*2*256, latent_size)  
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)  

    def forward(self, x):  
        """Extrai características e gera os parâmetros da distribuição latente.  

        Args:  
            x (Tensor): Imagem de entrada.  

        Returns:  
            Tuple[Tensor, Tensor]: Média (mu) e log da variância (logsigma).  
        """  
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))  
        x = F.relu(self.conv4(x))  
        x = x.view(x.size(0), -1)  

        mu = self.fc_mu(x)  
        logsigma = self.fc_logsigma(x)  
        return mu, logsigma  


class VAE(nn.Module):  
    """Variational Autoencoder (VAE).  

    Combina um codificador (Encoder) para comprimir imagens e um decodificador (Decoder) para reconstruí-las.  

    Attributes:  
        encoder (Encoder): Instância do codificador.  
        decoder (Decoder): Instância do decodificador.  
    """  

    def __init__(self, img_channels, latent_size):  
        """Inicializa o VAE com codificador e decodificador."""  
        super(VAE, self).__init__()  
        self.encoder = Encoder(img_channels, latent_size)  
        self.decoder = Decoder(img_channels, latent_size)  

    def forward(self, x):  
        """Passa os dados de entrada pelo VAE para codificação e reconstrução.  

        Args:  
            x (Tensor): Imagem de entrada.  

        Returns:  
            Tuple[Tensor, Tensor, Tensor]: Imagem reconstruída, média e log da variância.  
        """  
        mu, logsigma = self.encoder(x)  
        sigma = logsigma.exp()  
        eps = torch.randn_like(sigma)  
        z = eps.mul(sigma).add_(mu)  

        recon_x = self.decoder(z)  
        return recon_x, mu, logsigma  
