# CrossExit NIDS Pipeline

Este projeto é um pipeline completo de Sistema de Detecção de Intrusão em Rede (NIDS) baseado em Aprendizado de Máquina. O pipeline recebe capturas de tráfego bruto (arquivos PCAP), extrai fluxos de tráfego formatados em NetFlow v9 e realiza a classificação desses fluxos (Benigno vs. Ataque) usando uma rede neural com saídas antecipadas (Early Exits), conhecida como **IDSBranchyNet**.

---

## Como o Projeto Funciona

O pipeline opera em 5 etapas principais:

```
[PCAP de Entrada] 
       │
       ▼ (Passo 1: Divisão com editcap)
[Chunks de PCAP]
       │
       ▼ (Passo 2: Extração paralela com nProbe)
[Arquivos de Dump .flows]
       │
       ▼ (Passo 3: Consolidação e Seleção de Features)
[Dataframe de Features (32 colunas)]
       │
       ▼ (Passo 4: Inferência IDSBranchyNet)
[Predições / Rota de Saída]
       │
       ▼ (Passo 5: Escrita do Relatório)
[CSV de Saída Consolidado]
```

### 1. Divisão do PCAP (`editcap`)
Como o extrator de fluxos (`nProbe`) está rodando em modo de demonstração (limitado ao processamento de até 512 fluxos por execução), arquivos PCAP muito grandes são automaticamente fatiados em blocos menores (chunks) contendo até 2000 pacotes cada, utilizando a ferramenta `editcap`.

### 2. Extração de Fluxos Paralela (`nProbe`)
O pipeline inicializa um pool de threads (`ThreadPoolExecutor`) limitado dinamicamente ao número de núcleos de CPU disponíveis no host. Cada thread executa uma instância do `nProbe` sobre um chunk do PCAP, gerando arquivos temporários com fluxos NetFlow v9 estruturados e delimitados por um caractere separador (`#`).

### 3. Consolidação e Seleção de Características
Os fluxos parciais gerados por cada thread do `nProbe` são concatenados em um único DataFrame. O pipeline seleciona e ordena exatamente **32 características de rede** exigidas pelo modelo de detecção de intrusão (ex: quantidade de pacotes e bytes de entrada/saída, tempos de chegada entre pacotes - IAT, tipos de ICMP, portas e protocolos).

### 4. Normalização e Inferência com Early Exits (IDSBranchyNet)
Os dados são normalizados com um MinMaxScaler pré-treinado (`minmax_scaler.pkl`) e submetidos ao modelo PyTorch `IDSBranchyNet`. 

#### O Conceito de Early Exits (BranchyNet)
Diferente de redes neurais tradicionais que processam toda a sua profundidade para qualquer amostra, uma arquitetura de Early Exits insere ramificações de decisão intermediárias ao longo da rede:
* **Saída 1 (Exit 1)**: Uma ramificação rasa e rápida. Se a confiança da predição superar o limiar estabelecido (`t_atk1` para ataque, `t_norm1` para normal), o fluxo é classificado imediatamente, economizando tempo de CPU e latência.
* **Saída 2 (Exit 2)**: Se a confiança da Saída 1 for baixa, o fluxo continua sendo processado pelas camadas mais profundas da rede até a Saída 2, que realiza uma classificação mais complexa e robusta.
* **Rejeição**: Caso a confiança em ambas as saídas seja inferior aos limites configurados, o fluxo é marcado como **rejeitado** (valor `-1`), indicando que o tráfego é suspeito ou incerto para uma classificação automatizada segura.

### 5. Geração de Relatório
O relatório final é gravado no CSV de saída especificado, adicionando três novas colunas com os resultados da inteligência artificial:
* `PREDICAO_CLASSE`: `0` para Benigno, `1` para Ataque.
* `GRAU_CONFIANCA`: O nível de certeza do modelo (0.0 a 1.0).
* `ROTA_SAIDA_EXIT`: A saída que tomou a decisão (`1` para Saída 1, `2` para Saída 2, `-1` para Rejeitado/Incerto).

---

## Estrutura do Projeto

* `README.md`: Instruções gerais do projeto.
* `scripts/nids_pipeline.py`: Script principal de orquestração em Python.
* `scripts/Dockerfile`: Dockerfile contendo dependências de sistema (nProbe, editcap) e bibliotecas de ML (PyTorch, Pandas, Joblib).
* `scripts/docker-compose.yml`: Orquestração de volumes e configurações de execução do container.
* `scripts/models/`: Diretório contendo os artefatos de IA:
  * `model.pth`: Pesos pré-treinados do modelo PyTorch.
  * `minmax_scaler.pkl`: Objeto scaler para normalização dos dados.
* `scripts/data/`: Pasta compartilhada destinada a armazenar os PCAPs de entrada e relatórios de saída.

---

## Como Rodar o Projeto

Toda a execução do pipeline está encapsulada dentro de um container Docker, facilitando a portabilidade e evitando a necessidade de instalar localmente o `nProbe` ou o `PyTorch`.

### Pré-requisitos
* **Docker** instalado no host OS.
* **Docker Compose** instalado.

---

### Passo a Passo para Execução

#### 1. Preparar o Ambiente
Certifique-se de que a estrutura de diretórios em `scripts/` contém a pasta `models/` com os artefatos (`model.pth` e `minmax_scaler.pkl`) e a pasta `data/` criada.

#### 2. Adicionar o PCAP para Análise
Coloque o arquivo `.pcap` que você deseja analisar dentro do diretório `scripts/data/` no seu host. 
*(Exemplo: `scripts/data/meu_trafego.pcap`)*.

#### 3. Construir a Imagem Docker
Navegue até a pasta `scripts/` e execute o build da imagem:
```bash
cd scripts
docker compose build
```

#### 4. Executar o Pipeline
Rode o container passando o arquivo PCAP de entrada e o nome do arquivo CSV de saída desejado através das variáveis de ambiente `PCAP_INPUT` e `CSV_OUTPUT` (caminhos absolutos mapeados dentro do container sob a pasta `/app/data/`):

```bash
docker compose run --rm \
  -e PCAP_INPUT=/app/data/meu_trafego.pcap \
  -e CSV_OUTPUT=/app/data/relatorio_final.csv \
  cross-exit-nids
```

*Nota: `/app/data/` no container aponta diretamente para a pasta `scripts/data/` do host devido ao mapeamento de volume.*

#### 5. Analisar o Output e Métricas
Ao final do processamento, a tela exibirá as métricas consolidadas diretamente no terminal:

```text
2026-06-13 18:42:40,000 [INFO] Total de 363 fluxos consolidados de 2 parte(s).
2026-06-13 18:42:40,000 [INFO] Limpando arquivos temporários do nProbe...
2026-06-13 18:42:40,135 [INFO] Pesos do modelo carregados com sucesso.
2026-06-13 18:42:41,022 [INFO] Normalizador carregado com sucesso.
2026-06-13 18:42:41,025 [INFO] Executando inferência em lote para 363 fluxos extraídos...
2026-06-13 18:42:41,132 [INFO] Pipeline concluído com sucesso! Resultados salvos em: /app/data/relatorio_final.csv
2026-06-13 18:42:41,132 [INFO] Métricas de Classificação
2026-06-13 18:42:41,132 [INFO] Fluxos rejeitados: 215
2026-06-13 18:42:41,132 [INFO] Fluxos benignos na Saída 1: 0
2026-06-13 18:42:41,132 [INFO] Fluxos benignos na Saída 2: 1
2026-06-13 18:42:41,132 [INFO] Ataques na Saída 1: 10
2026-06-13 18:42:41,132 [INFO] Ataques na Saída 2: 137
```

O arquivo final `relatorio_final.csv` estará disponível em `scripts/data/relatorio_final.csv` no seu sistema de arquivos local para exploração.
