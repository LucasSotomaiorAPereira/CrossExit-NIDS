import sys
import json
import requests
import numpy as np
import joblib

# URL pública da sua API de inferência hospedada no Hugging Face Spaces (Fase 2)
API_URL = "https://sotomaior-early-exit-nids-api.hf.space/predict"

# Ordem exata das 32 características numéricas esperadas pelo modelo treinado
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

# Dicionário de tradução: chaves JSON nativas geradas pelo GoFlow -> nomes das características do modelo
MAPEAMENTO_GOFLOW = {
    'Proto': 'PROTOCOL',
    'Bytes': 'IN_BYTES',
    'Packets': 'IN_PKTS',
    'Etype': 'PROTOCOL' # Fallback para segurança caso protocolo L4 falhe
    # Caso seu pipeline use características bidirecionais combinadas, os campos complementares 
    # serão sintetizados abaixo na lógica do parser.
}

def carregar_scaler(caminho_scaler="minmax_scaler.pkl"):
    """Carrega o MinMaxScaler exportado ao final do seu treinamento."""
    try:
        scaler = joblib.load(caminho_scaler)
        print(f"[*] Scaler carregado e instanciado com sucesso a partir de '{caminho_scaler}'.")
        return scaler
    except Exception as e:
        print(f"[-] Erro crítico: Não foi possível ler o arquivo '{caminho_scaler}'.")
        print(f"    Garanta que ele foi colocado no mesmo diretório do script. Detalhes: {e}")
        sys.exit(1)

MAP_PROTOCOLOS = {
    'HOPOPT': 0, 'ICMP': 1, 'IGMP': 2, 'GGP': 3, 'IPv4': 4, 'ST': 5, 'TCP': 6, 'CBT': 7, 'EGP': 8, 'IGP': 9,
    'BBN-RCC-MON': 10, 'NVP-II': 11, 'PUP': 12, 'ARGUS': 13, 'EMCON': 14, 'XNET': 15, 'CHAOS': 16, 'UDP': 17,
    'MUX': 18, 'DCN-MEAS': 19, 'HMP': 20, 'PRM': 21, 'XNS-IDP': 22, 'TRUNK-1': 23, 'TRUNK-2': 24, 'LEAF-1': 25,
    'LEAF-2': 26, 'RDP': 27, 'IRTP': 28, 'ISO-TP4': 29, 'NETBLT': 30, 'MFE-NSP': 31, 'MERIT-INP': 32, 'DCCP': 33,
    '3PC': 34, 'IDPR': 35, 'XTP': 36, 'DDP': 37, 'IDPR-CMTP': 38, 'TP++': 39, 'IL': 40, 'IPv6': 41, 'SDRP': 42,
    'IPv6-Route': 43, 'IPv6-Frag': 44, 'IDRP': 45, 'RSVP': 46, 'GRE': 47, 'DSR': 48, 'BNA': 49, 'ESP': 50,
    'AH': 51, 'I-NLSP': 52, 'SWIPE': 53, 'NARP': 54, 'MOBILE': 55, 'TLSP': 56, 'SKIP': 57, 'IPv6-ICMP': 58,
    'IPv6-NoNxt': 59, 'IPv6-Opts': 60, 'CFTP': 62, 'SAT-EXPAK': 64, 'KRYPTOLAN': 65, 'RVD': 66, 'IPPC': 67,
    'SAT-MON': 69, 'VISA': 70, 'IPCV': 71, 'CPNX': 72, 'CPHB': 73, 'WSN': 74, 'PVP': 75, 'BR-SAT-MON': 76,
    'SUN-ND': 77, 'WB-MON': 78, 'WB-EXPAK': 79, 'ISO-IP': 80, 'VMTP': 81, 'SECURE-VMTP': 82, 'VINES': 83,
    'TTP': 84, 'IPTM': 84, 'NSFNET-IGP': 85, 'DGP': 86, 'TCF': 87, 'OSPF': 89, 'Sprite-RPC': 90, 'LARP': 91,
    'MTP': 92, 'AX.25': 93, 'IPIP': 94, 'MICP': 95, 'SCC-SP': 96, 'ETHERIP': 97, 'ENCAP': 98, 'GMTP': 100,
    'IFMP': 101, 'PNNI': 102, 'PIM': 103, 'ARIS': 104, 'SCPS': 105, 'QNX': 106, 'A/N': 107, 'IPComp': 108,
    'SNP': 109, 'Compaq-Peer': 110, 'IPX-in-IP': 111, 'VRRP': 112, 'PGM': 113, 'L2TP': 115, 'DDX': 116,
    'IATP': 117, 'STP': 118, 'SRP': 119, 'UTI': 120, 'SMP': 121, 'SM': 122, 'PTP': 123, 'ISIS over IPv4': 124,
    'FIRE': 125, 'CRTP': 126, 'CRUDP': 127, 'SSCOPMCE': 128, 'IPLT': 129, 'SPS': 130, 'PIPE': 131, 'SCTP': 132,
    'FC': 133, 'RSVP-E2E-IGNORE': 134, 'Mobility Header': 135, 'UDPLite': 136, 'MPLS-in-IP': 137, 'manet': 138,
    'HIP': 139, 'Shim6': 140, 'WESP': 141, 'ROHC': 142, 'Ethernet': 143
}

def converter_protocolo(valor):
    if isinstance(valor, (int, float)):
        return int(valor)
    if isinstance(valor, str):
        if valor.isdigit():
            return int(valor)
        return MAP_PROTOCOLOS.get(valor.upper(), 0)
    return 0

def mapear_e_filtrar_json(fluxo_bruto):
    """
    Consome o JSON bruto do GoFlow, traduz as métricas de rede para as chaves do modelo
    e monta o vetor numérico na ordem exata de entrada da rede neural.
    """
    # Normalização de chaves: garante suporte a JSON gerado tanto em snake_case quanto em CamelCase
    fluxo = {}
    mapeamento_chaves = {
        'proto': 'Proto',
        'bytes': 'Bytes',
        'packets': 'Packets',
        'etype': 'Etype',
        'time_flow_start': 'TimeFlowStart',
        'time_flow_end': 'TimeFlowEnd',
        'time_flow_start_ms': 'TimeFlowStartMs',
        'time_flow_end_ms': 'TimeFlowEndMs',
        'time_flow_start_ns': 'TimeFlowStartNs',
        'time_flow_end_ns': 'TimeFlowEndNs',
        'icmp_type': 'IcmpType',
        'icmp_code': 'IcmpCode',
        'src_addr': 'SrcAddr',
        'dst_addr': 'DstAddr',
        'src_port': 'SrcPort',
        'dst_port': 'DstPort'
    }
    for k, v in fluxo_bruto.items():
        if k in mapeamento_chaves:
            fluxo[mapeamento_chaves[k]] = v
        else:
            fluxo[k] = v

    # Cria uma cópia normalizada dos dados recebidos
    dados_normalizados = {}
    
    # 1. Realiza o mapeamento direto das chaves suportadas nativamente pelo GoFlow
    for chave_goflow, valor in fluxo.items():
        if chave_goflow in MAPEAMENTO_GOFLOW:
            chave_modelo = MAPEAMENTO_GOFLOW[chave_goflow]
            dados_normalizados[chave_modelo] = valor

    # Tratamento específico para o protocolo: converte strings ("TCP", "UDP") para seus IDs numéricos IANA
    protocolo_bruto = dados_normalizados.get('PROTOCOL', 0)
    dados_normalizados['PROTOCOL'] = converter_protocolo(protocolo_bruto)

    # 2. Lógica de preenchimento complementar / Simulação de tráfego bidirecional
    # Como o GoFlow extrai dados unidirecionais por padrão, normalizamos o tráfego 
    # de entrada (IN) e de saída (OUT) usando os dados do fluxo correspondente.
    if 'IN_BYTES' in dados_normalizados and 'OUT_BYTES' not in dados_normalizados:
        dados_normalizados['OUT_BYTES'] = int(dados_normalizados['IN_BYTES'] * 0.15) # Estimativa estatística de ACK
    if 'IN_PKTS' in dados_normalizados and 'OUT_PKTS' not in dados_normalizados:
        dados_normalizados['OUT_PKTS'] = int(dados_normalizados['IN_PKTS'] * 0.10)
        
    # Tratamento para durações de fluxos
    time_start = fluxo.get('TimeFlowStart', 0)
    time_end = fluxo.get('TimeFlowEnd', 0)
    time_start_ms = fluxo.get('TimeFlowStartMs', 0)
    time_end_ms = fluxo.get('TimeFlowEndMs', 0)
    time_start_ns = fluxo.get('TimeFlowStartNs', 0)
    time_end_ns = fluxo.get('TimeFlowEndNs', 0)
    
    if time_start_ns > 0 and time_end_ns > 0:
        duration_ms = float(time_end_ns - time_start_ns) / 1000000.0
        duration_sec = duration_ms / 1000.0
    elif time_start_ms > 0 and time_end_ms > 0:
        duration_ms = float(time_end_ms - time_start_ms)
        duration_sec = duration_ms / 1000.0
    else:
        duration_sec = float(time_end - time_start)
        duration_ms = duration_sec * 1000.0
        
    dados_normalizados['DURATION_IN'] = duration_sec
    dados_normalizados['DURATION_OUT'] = duration_sec
    dados_normalizados['FLOW_DURATION_MILLISECONDS'] = duration_ms

    # Estimação do tamanho de pacote e maior tamanho de pacote (LONGEST_FLOW_PKT, MAX_IP_PKT_LEN)
    in_bytes = dados_normalizados.get('IN_BYTES', 0)
    in_pkts = dados_normalizados.get('IN_PKTS', 0)
    avg_pkt_len = 0.0
    if in_bytes > 0 and in_pkts > 0:
        avg_pkt_len = float(in_bytes) / float(in_pkts)
        
    longest_pkt = avg_pkt_len
    if in_pkts > 1:
        # Se há múltiplos pacotes, o maior tende a se aproximar do MTU padrão de 1500 bytes para fluxos de dados
        longest_pkt = min(1500.0, avg_pkt_len * 1.2)
        
    dados_normalizados['LONGEST_FLOW_PKT'] = longest_pkt
    dados_normalizados['MAX_IP_PKT_LEN'] = longest_pkt

    # Estimação de throughput (bits por segundo - bps)
    if duration_sec > 0:
        dados_normalizados['SRC_TO_DST_AVG_THROUGHPUT'] = float(in_bytes * 8) / duration_sec
        dados_normalizados['DST_TO_SRC_AVG_THROUGHPUT'] = float(dados_normalizados.get('OUT_BYTES', 0) * 8) / duration_sec
    else:
        dados_normalizados['SRC_TO_DST_AVG_THROUGHPUT'] = 0.0
        dados_normalizados['DST_TO_SRC_AVG_THROUGHPUT'] = 0.0

    # Classificação de pacotes em bins de tamanho
    dados_normalizados['NUM_PKTS_UP_TO_128_BYTES'] = 0.0
    dados_normalizados['NUM_PKTS_128_TO_256_BYTES'] = 0.0
    dados_normalizados['NUM_PKTS_256_TO_512_BYTES'] = 0.0
    dados_normalizados['NUM_PKTS_512_TO_1024_BYTES'] = 0.0
    dados_normalizados['NUM_PKTS_1024_TO_1514_BYTES'] = 0.0
    
    if in_pkts > 0:
        if avg_pkt_len <= 128:
            dados_normalizados['NUM_PKTS_UP_TO_128_BYTES'] = float(in_pkts)
        elif avg_pkt_len <= 256:
            dados_normalizados['NUM_PKTS_128_TO_256_BYTES'] = float(in_pkts)
        elif avg_pkt_len <= 512:
            dados_normalizados['NUM_PKTS_256_TO_512_BYTES'] = float(in_pkts)
        elif avg_pkt_len <= 1024:
            dados_normalizados['NUM_PKTS_512_TO_1024_BYTES'] = float(in_pkts)
        else:
            dados_normalizados['NUM_PKTS_1024_TO_1514_BYTES'] = float(in_pkts)

    # Tratamento de campos ICMP
    icmp_type = fluxo.get('IcmpType', 0)
    icmp_code = fluxo.get('IcmpCode', 0)
    dados_normalizados['ICMP_TYPE'] = float((icmp_type * 256) + icmp_code)
    dados_normalizados['ICMP_IPV4_TYPE'] = float(icmp_type)

    # Estimação de Inter-Arrival Times (IAT) em milissegundos
    if in_pkts > 1:
        iat_avg = duration_ms / (in_pkts - 1)
        dados_normalizados['SRC_TO_DST_IAT_MIN'] = iat_avg * 0.8
        dados_normalizados['SRC_TO_DST_IAT_MAX'] = iat_avg * 1.2
        dados_normalizados['SRC_TO_DST_IAT_AVG'] = iat_avg
        dados_normalizados['SRC_TO_DST_IAT_STDDEV'] = iat_avg * 0.1
    else:
        dados_normalizados['SRC_TO_DST_IAT_MIN'] = 0.0
        dados_normalizados['SRC_TO_DST_IAT_MAX'] = 0.0
        dados_normalizados['SRC_TO_DST_IAT_AVG'] = 0.0
        dados_normalizados['SRC_TO_DST_IAT_STDDEV'] = 0.0

    out_pkts = dados_normalizados.get('OUT_PKTS', 0)
    if out_pkts > 1:
        iat_avg_out = duration_ms / (out_pkts - 1)
        dados_normalizados['DST_TO_SRC_IAT_MIN'] = iat_avg_out * 0.8
        dados_normalizados['DST_TO_SRC_IAT_MAX'] = iat_avg_out * 1.2
        dados_normalizados['DST_TO_SRC_IAT_AVG'] = iat_avg_out
        dados_normalizados['DST_TO_SRC_IAT_STDDEV'] = iat_avg_out * 0.1
    else:
        dados_normalizados['DST_TO_SRC_IAT_MIN'] = 0.0
        dados_normalizados['DST_TO_SRC_IAT_MAX'] = 0.0
        dados_normalizados['DST_TO_SRC_IAT_AVG'] = 0.0
        dados_normalizados['DST_TO_SRC_IAT_STDDEV'] = 0.0

    # 3. Monta o vetor ordenado garantindo que características ausentes não quebrem a inferência
    vetor_caracteristicas = []
    for nome_coluna in CARACTERISTICAS_MODELO:
        valor = dados_normalizados.get(nome_coluna, 0.0)
        vetor_caracteristicas.append(float(valor))
        
    return vetor_caracteristicas

def enviar_para_api(vetor_normalizado, metadados_fluxo):
    """Despacha o payload via HTTP POST para o endpoint FastAPI na Nuvem."""
    payload = {
        "features": vetor_normalizado.tolist()
    }
    
    try:
        resposta = requests.post(API_URL, json=payload, timeout=4)
        if resposta.status_code == 200:
            resultado = resposta.json()
            
            classe = resultado.get("classe")       # "Normal", "Ataque" ou "Rejeitado"
            ramo = resultado.get("ramo_saida")     # Identificação do ramo executor
            confianca = resultado.get("confianca", 0.0)
            
            # Formatação visual dos logs e alertas locais para o administrador do NIDS
            if classe == "Ataque":
                print(f"[ALERTA DE INTRUSÃO] {obter_identidade_fluxo(metadados_fluxo)}")
                print(f"    Veredito: {classe.upper()} | Confiança: {confianca:.4f} | Ramo: {ramo} | Confiança Máxima: {confianca:.4f}\n")
            elif classe == "Rejeitado":
                print(f"[TRÁFEGO SUSPEITO] {obter_identidade_fluxo(metadados_fluxo)}")
                print(f"    Ação: FLUXO REJEITADO (Incerteza Operacional) | Confiança Máxima: {confianca:.4f}\n")
            else:
                # Tráfego benigno classificado de forma limpa
                print(f"[Benigno] {obter_identidade_fluxo(metadados_fluxo)} -> Processado por: {ramo} | Confiança Máxima: {confianca:.4f}\n")
        else:
            print(f"[-] Erro de comunicação externa: Resposta HTTP {resposta.status_code} recebida do servidor de IA.")
            
    except requests.exceptions.RequestException as e:
        print(f"[-] Falha na requisição: API inacessível ou tempo limite esgotado. Detalhes: {e}")

def obter_identidade_fluxo(fluxo):
    """Extrai informações de endereçamento da camada de rede (IP e Portas) para fins de log local."""
    src_ip = fluxo.get('SrcAddr') or fluxo.get('src_addr', 'Desconhecido')
    dst_ip = fluxo.get('DstAddr') or fluxo.get('dst_addr', 'Desconhecido')
    src_port = fluxo.get('SrcPort') or fluxo.get('src_port', '?')
    dst_port = fluxo.get('DstPort') or fluxo.get('dst_port', '?')
    return f"{src_ip}:{src_port} -> {dst_ip}:{dst_port}"

def main():
    scaler = carregar_scaler()
    print("[*] Cliente Leve NIDS iniciado com sucesso.")
    print("[*] Aguardando fluxos decodificados vindos do pipeline do GoFlow...")
    
    # Processa de forma contínua o Standard Input (stdin) gerado pelo GoFlow
    for linha in sys.stdin:
        try:
            linha = linha.strip()
            if not linha:
                continue
                
            # Decodifica a string JSON enviada pelo pipe
            fluxo_json = json.loads(linha)
            
            # 1. Extrai, mapeia e padroniza as métricas
            vetor_bruto = mapear_e_filtrar_json(fluxo_json)
            
            # 2. Formata como matriz compatível com a entrada do scikit-learn (1, 32)
            dados_np = np.array(vetor_bruto).reshape(1, -1)
            
            # 3. Executa a transformação do Scaler estritamente baseada no histórico de treino
            dados_normalizados = scaler.transform(dados_np)[0]
            
            # 4. Envia o fluxo tratado para avaliação na nuvem
            enviar_para_api(dados_normalizados, fluxo_json)
            
        except json.JSONDecodeError:
            # Ignora fragmentos de pacotes ou linhas corrompidas no pipe
            continue
        except Exception as e:
            print(f"[-] Ocorreu um erro no processamento do fluxo atual: {e}")

if __name__ == "__main__":
    main()