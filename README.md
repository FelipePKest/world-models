# Implementacao e trabalho sobre artigo "WorldModels"

Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631. Para mais detalhes ver [github page](https://ctallec.github.io/world-models/).


## Configurações

Esse projeto requer o uso da linguagem Python3.10. 

Para executar os programas contidos nesse projeto, antes é necessário a instalacao das dependencias do mesmo. Isso pode ser feito atraves do ambiente Anaconda (https://www.anaconda.com/) ou através do pip. 
Para instalar as dependências, basta rodar o comando

```
pip3 install -r requirements.txt
```

em um terminal associado.

Esse projeto também faz o uso da biblioteca PyTorch. Para detalhes sobre sua instalação, visitar o site (https://pytorch.org)

## Executando o worldmodels

O modelo é composto por três partes:
  1. Um Autoencoder Variacional (VAE), cuja função é comprimir as imagens de entrada em uma representação latente compacta.
  2. Uma Rede Recorrente de Mistura de Densidade (MDN-RNN), treinada para prever a codificação latente do próximo quadro com base nas codificações latentes passadas e nas ações.
  3. Um Controlador Linear (C), que utiliza como entrada tanto a codificação latente do quadro atual quanto o estado oculto da MDN-RNN, considerando os latentes e ações passados, e gera uma ação como saída. Ele é treinado para maximizar a recompensa acumulada usando a Covariance-Matrix Adaptation Evolution Strategy (CMA-ES) do pacote cma em Python

Nesse projeto, as três seções são treinadas de maneira separada, usando os programas `trainvae.py`, `trainmdrnn.py` and `traincontroller.py`. 

Os scripts de treinamento recebem os seguintes argumentos:
* **--logdir** : O diretorio onde os modelos serão salvos. Caso os arquivos dos modelos ja existam, o treinamento resume do ponto onde foi parado.
* **--noreload** : Caso se deseja reiniciar um treinamento nos modelos localizados no *logdir*, basta configurar esse argumento como true .

### 1. Geracao dos dados

Antes de treinar os modelos VAE e MDN-RNN, é necessario gerar um dataset para servir como os dados a serem utilizados para o treino dos mesmos. Esses dados são armazenados no diretorio `datasets/carracing`.

Isso é feito através do script `data/generation_script.py`.

```bash
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```

Os dados são gerados a partir de uma politica de movimento *browniano*.

### 2. Treinando o VAE

Com os dados gerados, podemos comecar treinando o VAE, rodando o comando:
```bash
python trainvae.py --logdir exp_dir
```

### 3. Treinando o MDN-RNN
A MDN-RNN é treinada pelo arquivo `trainmdrnn.py`.
```bash
python trainmdrnn.py --logdir exp_dir
```
É necessario haver treinado um modelo VAE no mesmo diretório `exp_dir` para esse codigo funcionar adequadamente.

### 4. Treinando e testando o Controller
Por fim, um Controller é treinado pelo arquivo `traincontroller.py`, e.g.
```bash
python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```

Com o Controller treinado, pode-se testar a política treinada pelo arquivo `test_controller.py` e.g.
```bash
python test_controller.py --logdir exp_dir
```

# Referências

* **Corentin Tallec** - [ctallec](https://github.com/ctallec)
* **Léonard Blier** - [leonardblier](https://github.com/leonardblier)
* **Diviyan Kalainathan** - [diviyan-kalainathan](https://github.com/diviyan-kalainathan)

[1] Ha, D. and Schmidhuber, J. World Models, 2018

[2] Kingma, D., Auto-Encoding Variational Bayes, 2014

[3] Graves, A., Generating Sequences With Recurrent Neural Networks, 2013

[4] Hansen, N., The CMA evolution strategy: a comparing review, 2006

[5] Irpan, A., [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html), 2018

[6] Henderson, P. Islam, R. Bachman, P. Pineau, J. Precup, D. Meger, D., Deep Reinforcement Learning that Matters, 2017
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
