FROM python:3.10-slim

# Instala a biblioteca nativa do Linux (libpcap) que o Scapy precisa para capturar pacotes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho interno
WORKDIR /app

# Copia e instala as dependências do Python (incluindo o Scapy agora)
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia o agente de rede e o scaler
COPY traffic_agent.py /app/traffic_agent.py
COPY minmax_scaler.pkl /app/minmax_scaler.pkl

# Garante permissões de execução no agente Python
RUN chmod +x /app/traffic_agent.py

# Executa diretamente o script Python, sem YAF ou pipes intermediários
CMD ["python3", "/app/traffic_agent.py"]