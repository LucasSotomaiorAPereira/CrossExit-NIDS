# CrossExit-NIDS

A ferramenta CrossExit-NIDS foi desenvolvida com um pipeline de classificação para fluxos de tráfego de rede (PCAP) utilizando a arquitetura de rede neural com saídas antecipadas (Early Exits) e extração de características em tempo real com nProbe de forma paralela.


## Requisitos

### Clonando o Repositório

Primeiramente, clone o repositório em sua máquina local:

```sh
git clone https://github.com/LucasSotomaiorAPereira/CrossExit-NIDS.git
cd CrossExit-NIDS
```

### Docker e Dependências

Para executar este projeto, o Docker e o Docker Compose são as únicas dependências necessárias em sua máquina local. Todas as demais dependências (como os pacotes Python listados no `scripts/Dockerfile`) são configuradas e executadas de forma isolada dentro dos containers.

#### Instalando o Docker e Docker Compose

##### Linux (Ubuntu/Debian)
1. Instale o Docker e o Docker Compose seguindo as instruções no [site oficial do Docker](https://docs.docker.com/get-docker/).
2. Adicione seu usuário ao grupo Docker para evitar a necessidade de permissões de root/sudo ao rodar comandos docker:
    ```sh
    sudo usermod -aG docker $USER
    ```
3. Faça logout e login novamente, ou reinicie o sistema, para que as alterações tenham efeito.

##### Windows e macOS
1. Instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/), que já inclui o Docker Compose integrado.
2. No Windows, recomenda-se utilizar a integração com o WSL 2 (Windows Subsystem for Linux) para melhor performance.
3. Não há necessidade de configurar grupos ou permissões adicionais após a instalação típica do Docker Desktop.

##### Validando a Instalação (Qualquer SO)
Execute o comando abaixo no seu terminal (WSL, Terminal do macOS ou PowerShell/CMD no Windows) para verificar a instalação:
```sh
docker --version
docker compose version
```

##### Construindo a Imagem
Acesse o diretório `./scripts` e construa a imagem local necessária para a análise:
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
[INFO] PCAP dividido em 16 parte(s).
[INFO] Iniciando ThreadPoolExecutor com 8 workers para processar 16 partes...
[INFO] Processando parte 1/16
[INFO] Processando parte 2/16
[INFO] Processando parte 3/16
[INFO] Processando parte 4/16
[INFO] Processando parte 5/16
[INFO] Processando parte 6/16
[INFO] Processando parte 7/16
[INFO] Processando parte 8/16
[INFO] Processando parte 9/16
[INFO] Processando parte 10/16
[INFO] Processando parte 11/16
[INFO] Processando parte 12/16
[INFO] Processando parte 13/16
[INFO] Processando parte 14/16
[INFO] Processando parte 15/16
[INFO] Processando parte 16/16
[INFO] Total de 391 fluxos consolidados de 16 parte(s).
[INFO] Limpando arquivos temporários do nProbe...
[INFO] Pesos do modelo carregados com sucesso.
[INFO] Normalizador carregado com sucesso.
[INFO] Executando inferência em lote para 391 fluxos extraídos...
[INFO] Pipeline concluído com sucesso! Resultados salvos em: /app/data/output.csv
[INFO] Métricas de Classificação
[INFO] Fluxos rejeitados: 119
[INFO] Fluxos benignos na Saída 1: 17
[INFO] Fluxos benignos na Saída 2: 141
[INFO] Ataques na Saída 1: 16
[INFO] Ataques na Saída 2: 98
```

O arquivo final `output.csv` estará disponível em `scripts/data/output.csv` no sistema de arquivos local.

Na primeira execução da ferramenta, as dependências serão verificadas. Caso ocorra um erro, a ferramenta será abortada. Para corrigir as dependências veja os [requisitos](#requisitos).

Os dados informados a partir do argumento da aplicação são processados sob um diretório temporário `.temp_pcap_nprobe/` que é automaticamente limpo ao final do processamento.
