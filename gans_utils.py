# gans_utils.py
import torch.nn as nn

# Parâmetros da nossa GAN
LATENT_SIZE = 100  # Tamanho do vetor de ruído de entrada
CHANNELS_IMG = 1   # Canais da imagem (1 para MNIST, escala de cinza)
FEATURES_G = 64    # Fator de tamanho para o Gerador
FEATURES_D = 64    # Fator de tamanho para o Discriminador

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Entrada: Ruído (LATENT_SIZE)
            # Saída: Imagem (CHANNELS_IMG x 64 x 64)
            self._block(LATENT_SIZE, FEATURES_G * 8, 4, 1, 0),  # 4x4
            self._block(FEATURES_G * 8, FEATURES_G * 4, 4, 2, 1), # 8x8
            self._block(FEATURES_G * 4, FEATURES_G * 2, 4, 2, 1), # 16x16
            self._block(FEATURES_G * 2, FEATURES_G, 4, 2, 1),     # 32x32
            nn.ConvTranspose2d(FEATURES_G, CHANNELS_IMG, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Normaliza a saída para [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Entrada: Imagem (CHANNELS_IMG x 64 x 64)
            # Saída: Probabilidade (real ou falsa)
            nn.Conv2d(CHANNELS_IMG, FEATURES_D, 4, 2, 1), # 32x32
            nn.LeakyReLU(0.2),
            self._block(FEATURES_D, FEATURES_D * 2, 4, 2, 1),     # 16x16
            self._block(FEATURES_D * 2, FEATURES_D * 4, 4, 2, 1), # 8x8
            self._block(FEATURES_D * 4, FEATURES_D * 8, 4, 2, 1), # 4x4
            nn.Conv2d(FEATURES_D * 8, 1, 4, 1, 0), # 1x1
            nn.Sigmoid() # Probabilidade entre 0 e 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)