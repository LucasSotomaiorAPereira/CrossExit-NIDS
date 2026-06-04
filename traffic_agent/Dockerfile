# Etapa 1: Obter o binário oficial do goflow2
FROM netsampler/goflow2:latest AS goflow_base

# Etapa 2: Criar a imagem final com Python
FROM python:3.10-slim

# Instalar utilitários básicos do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar o binário do goflow2 para o path de execução do sistema
COPY --from=goflow_base /goflow2 /usr/local/bin/goflow2

# Copiar e instalar as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o script do agente e o scaler do MinMaxScaler
COPY traffic_agent.py .
COPY minmax_scaler.pkl .

# Expor a porta UDP padrão para receber tráfego NetFlow v9/IPFIX
EXPOSE 2055/udp

# Comando Inicial: Executa o goflow2 e canaliza (pipe '|') a saída JSON 
# em tempo real diretamente para o Standard Input (stdin) do Python.
# O parâmetro '-u' força o Python a processar as linhas sem buffering.
CMD ["sh", "-c", "goflow2 -listen netflow://0.0.0.0:2055 -format json | python -u traffic_agent.py"]