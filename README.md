# Projeto CrossExit-NIDS

A ferramenta CrossExit-NIDS foi desenvolvida com um pipeline de classificação para fluxos de tráfego de rede (PCAP) utilizando a arquitetura de rede neural com saídas antecipadas (Early Exits) e extração de características em tempo real com nProbe de forma paralela.


## Requisitos

Para instalar e executar este projeto, é necessário ter o Docker instalado na máquina. Os pacotes Python necessários estão listados no arquivo `scripts/Dockerfile`.

### Instalando o Docker
1. Siga as instruções para instalar o Docker na sua máquina a partir do [site oficial do Docker](https://docs.docker.com/get-docker/).

2. Adicione seu usuário ao grupo Docker para evitar a necessidade de permissões root:
    ```sh
    sudo usermod -aG docker $USER
    ```

3. Faça logout e login novamente, ou reinicie o sistema, para que as alterações tenham efeito.

4. Verifique a instalação do Docker com o comando:
    ```sh
    docker --version
    ```

5. Construa a imagem local necessária para a análise:
    ```sh
    cd scripts
    docker compose build
    ```

### Outras dependências

Para o correto funcionamento dessa ferramenta, os seguintes arquivos devem estar presentes em `./scripts/models/`:
  - `model.pth` — pesos pré-treinados do modelo PyTorch.
  - `minmax_scaler.pkl` — objeto scaler para normalização dos dados.

### Estrutura de diretórios:

#### ./scripts/data/
- **Função:** Diretório onde o arquivo PCAP a ser analisado é armazenado e onde o relatório em CSV é salvo.

#### ./scripts/models/
- **Função:** Diretório contendo os arquivos de modelo e escala utilizados para normalização e previsão das features extraídas.

#### ./scripts/
- **Função:** Diretório com todos os arquivos fontes, configurações do Docker Compose e Dockerfile correspondente.

## Executando a Aplicação
A execução da aplicação pode ser feita de forma simples através do Docker Compose, passando os caminhos do PCAP de entrada e do CSV de saída. Por exemplo, consideremos o arquivo `./scripts/data/input.pcap` e o arquivo de saída `./scripts/data/output.csv`.

Acesse o diretório `./scripts` e execute o comando:

```sh
# para processar a captura e gerar o relatório final consolidado
docker compose run --rm -e PCAP_INPUT=/app/data/input.pcap -e CSV_OUTPUT=/app/data/output.csv cross-exit-nids
```

### Saída Esperada

Ao final do processamento, o terminal exibirá um log semelhante a este:

```text
[INFO] Dividindo PCAP em partes de 2000 pacotes cada...
[INFO] PCAP dividido em 2 parte(s).
[INFO] Iniciando ThreadPoolExecutor com 2 workers para processar 2 partes...
[INFO] Processando parte 1/2
[INFO] Processando parte 2/2
[INFO] Total de 363 fluxos consolidados de 2 parte(s).
[INFO] Limpando arquivos temporários do nProbe...
[INFO] Pesos do modelo carregados com sucesso.
[INFO] Normalizador carregado com sucesso.
[INFO] Executando inferência em lote para 363 fluxos extraídos...
[INFO] Pipeline concluído com sucesso! Resultados salvos em: /app/data/output.csv
[INFO] Métricas de Classificação
[INFO] Fluxos rejeitados: 215
[INFO] Fluxos benignos na Saída 1: 0
[INFO] Fluxos benignos na Saída 2: 1
[INFO] Ataques na Saída 1: 10
[INFO] Ataques na Saída 2: 137
```

O arquivo final `output.csv` estará disponível em `scripts/data/output.csv` no sistema de arquivos local.

Na primeira execução da ferramenta, as dependências serão verificadas. Caso ocorra um erro, a ferramenta será abortada. Para corrigir as dependências veja os [requisitos](#requisitos).

Os dados informados a partir do argumento da aplicação são processados sob um diretório temporário `.temp_pcap_nprobe/` que é automaticamente limpo ao final do processamento.
