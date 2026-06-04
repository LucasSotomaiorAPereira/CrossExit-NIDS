import time
import pickle
import threading
import numpy as np
from scapy.all import sniff, IP, TCP, UDP, ICMP, DNS
import requests

API_URL = "https://sotomaior-early-exit-nids-api.hf.space/predict"

# ==========================================
# CONFIGURAÇÕES E ORDEM ESTRETA DAS FEATURES
# ==========================================
CARACTERISTICAS_MODELO = [
                            'PROTOCOL',
                            'IN_BYTES',
                            'IN_PKTS',
                            'OUT_BYTES',
                            'OUT_PKTS',
                            'FLOW_DURATION_MILLISECONDS',
                            'DURATION_IN',
                            'DURATION_OUT',
                            'LONGEST_FLOW_PKT',
                            'MAX_IP_PKT_LEN',
                            'RETRANSMITTED_IN_BYTES',
                            'RETRANSMITTED_IN_PKTS',
                            'RETRANSMITTED_OUT_BYTES',
                            'RETRANSMITTED_OUT_PKTS',
                            'SRC_TO_DST_AVG_THROUGHPUT',
                            'DST_TO_SRC_AVG_THROUGHPUT',
                            'NUM_PKTS_UP_TO_128_BYTES',
                            'NUM_PKTS_128_TO_256_BYTES',
                            'NUM_PKTS_256_TO_512_BYTES',
                            'NUM_PKTS_512_TO_1024_BYTES',
                            'NUM_PKTS_1024_TO_1514_BYTES',
                            'ICMP_TYPE',
                            'ICMP_IPV4_TYPE',
                            'DNS_QUERY_TYPE',
                            'SRC_TO_DST_IAT_MIN',
                            'SRC_TO_DST_IAT_MAX',
                            'SRC_TO_DST_IAT_AVG',
                            'SRC_TO_DST_IAT_STDDEV',
                            'DST_TO_SRC_IAT_MIN',
                            'DST_TO_SRC_IAT_MAX',
                            'DST_TO_SRC_IAT_AVG',
                            'DST_TO_SRC_IAT_STDDEV',
                        ]

# Cache global de fluxos ativos em memória
fluxos_ativos = {}
LOCK_FLUXOS = threading.Lock()
FLOW_TIMEOUT = 15.0  # Expira fluxos inativos após 15 segundos

# Carrega o Scaler do seu modelo
try:
    with open('minmax_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("[+] Scaler 'minmax_scaler.pkl' carregado com sucesso.")
except Exception as e:
    print(f"[-] Erro ao carregar Scaler: {e}. Usando fallback sem normalização.")
    scaler = None


class NetworkFlow:
    """Classe responsável por acumular e calcular o estado estatístico de um Biflow."""
    def __init__(self, src_ip, sport, dst_ip, dport, proto):
        self.src_ip = src_ip
        self.sport = sport
        self.dst_ip = dst_ip
        self.dport = dport
        self.proto = proto
        
        # Timestamps
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.ts_in = []
        self.ts_out = []
        
        # Volumetria
        self.in_bytes = 0
        self.in_pkts = 0
        self.out_bytes = 0
        self.out_pkts = 0
        
        # Tamanhos
        self.longest_pkt = 0
        self.max_ip_pkt_len = 0
        self.pkt_bins = [0, 0, 0, 0, 0] # Up to 128, 256, 512, 1024, 1514
        
        # Camada de Transporte / Aplicação
        self.seen_seqs_in = set()
        self.seen_seqs_out = set()
        self.retransmitted_in_bytes = 0
        self.retransmitted_in_pkts = 0
        self.retransmitted_out_bytes = 0
        self.retransmitted_out_pkts = 0
        
        self.icmp_type = 0
        self.icmp_ipv4_type = 0
        self.dns_query_type = 0

    def atualizar(self, pkt, direcao_inbound):
        self.last_seen = time.time()
        pkt_len = len(pkt)
        
        ip_len = pkt[IP].len if pkt.haslayer(IP) else pkt_len
        
        if pkt_len > self.longest_pkt: self.longest_pkt = pkt_len
        if ip_len > self.max_ip_pkt_len: self.max_ip_pkt_len = ip_len
        
        if ip_len <= 128: self.pkt_bins[0] += 1
        elif ip_len <= 256: self.pkt_bins[1] += 1
        elif ip_len <= 512: self.pkt_bins[2] += 1
        elif ip_len <= 1024: self.pkt_bins[3] += 1
        elif ip_len <= 1514: self.pkt_bins[4] += 1   

        if pkt.haslayer(ICMP):
            self.icmp_ipv4_type = int(pkt[ICMP].type)                          # só o tipo
            self.icmp_type = self.icmp_ipv4_type * 256 + int(pkt[ICMP].code)  # combinado
            
        if pkt.haslayer(DNS) and pkt[DNS].qd:
            self.dns_query_type = pkt[DNS].qd.qtype

        if direcao_inbound:
            self.in_pkts += 1
            self.in_bytes += ip_len
            self.ts_in.append(self.last_seen)
            
            if pkt.haslayer(TCP) and len(pkt[TCP].payload) > 0:
                seq = pkt[TCP].seq
                if seq in self.seen_seqs_in:
                    self.retransmitted_in_pkts += 1
                    self.retransmitted_in_bytes += ip_len
                self.seen_seqs_in.add(seq)
        else:
            self.out_pkts += 1
            self.out_bytes += ip_len
            self.ts_out.append(self.last_seen)
            
            if pkt.haslayer(TCP) and len(pkt[TCP].payload) > 0:
                seq = pkt[TCP].seq
                if seq in self.seen_seqs_out:
                    self.retransmitted_out_pkts += 1
                    self.retransmitted_out_bytes += ip_len
                self.seen_seqs_out.add(seq)

    def calcular_estatisticas_iat(self, timestamps):
        if len(timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0
        iats = [ (timestamps[i] - timestamps[i-1]) * 1000.0 for i in range(1, len(timestamps)) ] # Convertido para ms
        return float(np.min(iats)), float(np.max(iats)), float(np.mean(iats)), float(np.std(iats))

    def exportar_vetor_features(self):
        """Compila os dados acumulados estritamente na ordem exigida pelo modelo."""
        duracao_total_ms = (self.last_seen - self.first_seen) * 1000.0
        duracao_in = (self.ts_in[-1] - self.ts_in[0]) * 1000.0 if self.ts_in else 0.0
        duracao_out = (self.ts_out[-1] - self.ts_out[0]) * 1000.0 if self.ts_out else 0.0
        
        # Throughput (Bytes por segundo)
        duracao_total_sec = (self.last_seen - self.first_seen)
        throughput_in = (self.in_bytes / duracao_total_sec) * 8.0 if duracao_total_sec > 0 else 0.0
        throughput_out = (self.out_bytes / duracao_total_sec) * 8.0 if duracao_total_sec > 0 else 0.0
        
        # Cálculos de IAT
        iat_in_min, iat_in_max, iat_in_avg, iat_in_std = self.calcular_estatisticas_iat(self.ts_in)
        iat_out_min, iat_out_max, iat_out_avg, iat_out_std = self.calcular_estatisticas_iat(self.ts_out)
        
        features = {
            'PROTOCOL': float(self.proto),
            'IN_BYTES': float(self.in_bytes),
            'IN_PKTS': float(self.in_pkts),
            'OUT_BYTES': float(self.out_bytes),
            'OUT_PKTS': float(self.out_pkts),
            'FLOW_DURATION_MILLISECONDS': float(duracao_total_ms),
            'DURATION_IN': float(duracao_in),
            'DURATION_OUT': float(duracao_out),
            'LONGEST_FLOW_PKT': float(self.longest_pkt),
            'MAX_IP_PKT_LEN': float(self.max_ip_pkt_len),
            'RETRANSMITTED_IN_BYTES': float(self.retransmitted_in_bytes),
            'RETRANSMITTED_IN_PKTS': float(self.retransmitted_in_pkts),
            'RETRANSMITTED_OUT_BYTES': float(self.retransmitted_out_bytes),
            'RETRANSMITTED_OUT_PKTS': float(self.retransmitted_out_pkts),
            'SRC_TO_DST_AVG_THROUGHPUT': float(throughput_in),
            'DST_TO_SRC_AVG_THROUGHPUT': float(throughput_out),
            'NUM_PKTS_UP_TO_128_BYTES': float(self.pkt_bins[0]),
            'NUM_PKTS_128_TO_256_BYTES': float(self.pkt_bins[1]),
            'NUM_PKTS_256_TO_512_BYTES': float(self.pkt_bins[2]),
            'NUM_PKTS_512_TO_1024_BYTES': float(self.pkt_bins[3]),
            'NUM_PKTS_1024_TO_1514_BYTES': float(self.pkt_bins[4]),
            'ICMP_TYPE': float(self.icmp_type),
            'ICMP_IPV4_TYPE': float(self.icmp_ipv4_type),
            'DNS_QUERY_TYPE': float(self.dns_query_type),
            'SRC_TO_DST_IAT_MIN': iat_in_min,
            'SRC_TO_DST_IAT_MAX': iat_in_max,
            'SRC_TO_DST_IAT_AVG': iat_in_avg,
            'SRC_TO_DST_IAT_STDDEV': iat_in_std,
            'DST_TO_SRC_IAT_MIN': iat_out_min,
            'DST_TO_SRC_IAT_MAX': iat_out_max,
            'DST_TO_SRC_IAT_AVG': iat_out_avg,
            'DST_TO_SRC_IAT_STDDEV': iat_out_std
        }
        
        # Garante o alinhamento de vetor estrito para a entrada do modelo
        return [features[col] for col in CARACTERISTICAS_MODELO]


def processar_pacote(pkt):
    """Callback invocado para cada frame capturado na interface de rede."""
    if not pkt.haslayer(IP):
        return  # Ignora tráfego que não seja IP (ex: ARP brutos, STP)

    proto = pkt[IP].proto
    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    
    # Determinação de portas baseada na camada de transporte
    if pkt.haslayer(TCP):
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif pkt.haslayer(UDP):
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        sport, dport = 0, 0  # Fallback para ICMP/outros sem porta L4

    # Heurística de Chave Bidirecional (Casamento Cliente -> Servidor)
    # Procuramos se o fluxo inverso já existe na tabela de memória
    chave_direta = (proto, src_ip, sport, dst_ip, dport)
    chave_reversa = (proto, dst_ip, dport, src_ip, sport)
    
    with LOCK_FLUXOS:
        if chave_direta in fluxos_ativos:
            fluxos_ativos[chave_direta].atualizar(pkt, direcao_inbound=True)
        elif chave_reversa in fluxos_ativos:
            fluxos_ativos[chave_reversa].atualizar(pkt, direcao_inbound=False)
        else:
            # Novo fluxo detectado. O primeiro pacote determina a direção padrão do cliente (Inbound)
            novo_fluxo = NetworkFlow(src_ip, sport, dst_ip, dport, proto)
            novo_fluxo.atualizar(pkt, direcao_inbound=True)
            fluxos_ativos[chave_direta] = novo_fluxo

    # Se for um encerramento explícito de conexão TCP, podemos forçar uma classificação antecipada
    # if pkt.haslayer(TCP) and any(f in pkt[TCP].flags for f in ['F', 'R']):
    #     enviar_para_inferencia(chave_direta if chave_direta in fluxos_ativos else chave_reversa)


def enviar_para_inferencia(chave_fluxo):
    """Exporta, normaliza e submete o vetor de características ao Modelo Inteligente no Hugging Face."""
    with LOCK_FLUXOS:
        fluxo = fluxos_ativos.pop(chave_fluxo, None)
        
    if fluxo and (fluxo.in_pkts + fluxo.out_pkts) > 1:
        vetor_bruto = fluxo.exportar_vetor_features()
        
        # Preparação matemática (Reshape 1D -> 2D)
        dados_input = np.array(vetor_bruto).reshape(1, -1)
        
        if scaler:
            dados_input = scaler.transform(dados_input)
            
        print(f"\n[*] Fluxo Detectado: {fluxo.src_ip}:{fluxo.sport} -> {fluxo.dst_ip}:{fluxo.dport} | Total Pkts: {fluxo.in_pkts + fluxo.out_pkts}")
        
        # 1. Converter o vetor NumPy para uma lista Python nativa (necessário para o JSON)
        vetor_lista = dados_input[0].tolist()
        
        # 2. Montar o payload conforme o formato esperado pelo seu backend FastAPI
        payload = {
            "features": vetor_lista
        }
            
        # 3. Disparar a requisição HTTP POST para o Hugging Face Space
        try:
            # Definimos um timeout curto (ex: 3s) para o agente não travar se a API demorar
            response = requests.post(API_URL, json=payload, timeout=3)
            
            if response.status_code == 200:
                resultado = response.json()
                
                # Extraindo os metadados científicos retornados pela sua API (Early Exit e Rejeição)
                predicao = resultado.get("prediction", "Desconhecido")
                ramo_exit = resultado.get("exit_branch", "Final")
                confianca = resultado.get("confidence", 0.0)
                
                # Formatação visual dos alertas no terminal do NIDS
                if "Ataque" in predicao or "Suspeito" in predicao:
                    status_prefix = "[ALERTA DE INTRUSÃO]"
                elif "Rejeitado" in predicao:
                    status_prefix = "[TRÁFEGO INCERTO / REJEITADO]"
                else:
                    status_prefix = "[BENIGNO]"
                    
                print(f"{status_prefix} Classificação: {predicao} | Ramo Utilizado: Ramo {ramo_exit} | Confiança: {confianca:.2%}")
                
            else:
                print(f"[-] Erro na API Hugging Face (Status {response.status_code}): {response.text}")
                
        except requests.exceptions.Timeout:
            print("[-] Timeout atingido! A API do Hugging Face demorou demais para responder.")
        except requests.exceptions.RequestException as e:
            print(f"[-] Falha crítica na comunicação com a API de inferência: {e}")


def monitor_de_expiracao():
    """Thread em background para limpar buffers de fluxos inativos ou de longa duração (Streaming)."""
    while True:
        time.sleep(5)
        agora = time.time()
        chaves_para_remover = []
        
        with LOCK_FLUXOS:
            for chave, fluxo in fluxos_ativos.items():
                if agora - fluxo.last_seen > FLOW_TIMEOUT:
                    chaves_para_remover.append(chave)
                    
        for chave in chaves_para_remover:
            enviar_para_inferencia(chave)


# ==========================================
# INICIALIZAÇÃO DO AGENTE DE REDE
# ==========================================
if __name__ == "__main__":
    import os
    interface_alvo = os.getenv("INTERFACE", "wlp2s0")
    print(f"[*] Iniciando Coletor Stateful baseado em Scapy Nativo na interface {interface_alvo}...")
    
    # Ativa o coletor periódico em background de fluxos inativos
    thread_garbage_collector = threading.Thread(target=monitor_de_expiracao, daemon=True)
    thread_garbage_collector.start()
    
    print("[*] Aguardando e capturando pacotes em modo promíscuo... Pressione Ctrl+C para encerrar.")
    # sniff() captura pacotes continuamente. 
    # store=0 impede acúmulo de pacotes na RAM física da máquina, vital para longa execução.
    sniff(iface=interface_alvo, filter="ip", prn=processar_pacote, store=0)