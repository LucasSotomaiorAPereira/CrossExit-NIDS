FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    wget \
    lsb-release \
    gnupg \
    whiptail \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://packages.ntop.org/apt/bookworm/all/apt-ntop.deb \
    && dpkg -i apt-ntop.deb \
    && rm apt-ntop.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends nprobe \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    pandas==2.2.1 \
    numpy==1.26.4 \
    joblib==1.3.2 \
    scikit-learn==1.7.2 \
    scapy==2.6.1 \
    requests==2.31.0

COPY nids_pipeline.py /app/nids_pipeline.py
COPY model.pth /app/model.pth
COPY minmax_scaler.pkl /app/minmax_scaler.pkl

RUN chmod +x /app/nids_pipeline.py

ENV PCAP_INPUT=/app/data/input.pcap
ENV CSV_OUTPUT=/app/data/output.csv

CMD ["sh", "-c", "python3 /app/nids_pipeline.py $PCAP_INPUT $CSV_OUTPUT"]