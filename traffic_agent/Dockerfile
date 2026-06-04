FROM certcc/silklive:latest

# Define o diretório de trabalho interno
WORKDIR /app

# Instala o gerenciador de pacotes do Python e as dependências do modelo
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir numpy requests joblib scikit-learn

# Copia o agente de rede refatorado e o scaler gerado no treino
COPY traffic_agent.py /app/traffic_agent.py
COPY minmax_scaler.pkl /app/minmax_scaler.pkl

# Garante permissões de execução no agente Python
RUN chmod +x /app/traffic_agent.py

# Variável de ambiente padrão para a interface de rede (pode ser sobrescrita no compose)
ENV INTERFACE=wlp2s0

# Inicializa o YAF em background (&) e passa o fluxo em tempo real para o Super Mediator + Python
CMD yaf --in $INTERFACE --live pcap --out 127.0.0.1 --ipfix tcp --ipfix-port 9999 --biflow --max-payload 2048 & \
    super_mediator --ipfix-port 9999 --out - --format JSON | python3 /app/traffic_agent.py