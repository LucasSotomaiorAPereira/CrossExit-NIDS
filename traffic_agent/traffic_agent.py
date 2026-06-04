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

def mapear_e_filtrar_json(fluxo_bruto):
    """
    Consome o JSON bruto do GoFlow, traduz as métricas de rede para as chaves do modelo
    e monta o vetor numérico na ordem exata de entrada da rede neural.
    """
    # Cria uma cópia normalizada dos dados recebidos
    dados_normalizados = {}
    
    # 1. Realiza o mapeamento direto das chaves suportadas nativamente pelo GoFlow
    for chave_goflow, valor in fluxo_bruto.items():
        if chave_goflow in MAPEAMENTO_GOFLOW:
            chave_modelo = MAPEAMENTO_GOFLOW[chave_goflow]
            dados_normalizados[chave_modelo] = valor

    # 2. Lógica de preenchimento complementar / Simulação de tráfego bidirecional
    # Como o GoFlow extrai dados unidirecionais por padrão, normalizamos o tráfego 
    # de entrada (IN) e de saída (OUT) usando os dados do fluxo correspondente.
    if 'IN_BYTES' in dados_normalizados and 'OUT_BYTES' not in dados_normalizados:
        dados_normalizados['OUT_BYTES'] = int(dados_normalizados['IN_BYTES'] * 0.15) # Estimativa estatística de ACK
    if 'IN_PKTS' in dados_normalizados and 'OUT_PKTS' not in dados_normalizados:
        dados_normalizados['OUT_PKTS'] = int(dados_normalizados['IN_PKTS'] * 0.10)
        
    # Tratamento para durações de fluxos
    time_start = fluxo_bruto.get('TimeFlowStart', 0)
    time_end = fluxo_bruto.get('TimeFlowEnd', 0)
    time_start_ms = fluxo_bruto.get('TimeFlowStartMs', 0)
    time_end_ms = fluxo_bruto.get('TimeFlowEndMs', 0)
    
    if time_start_ms > 0 and time_end_ms > 0:
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
    icmp_type = fluxo_bruto.get('IcmpType', 0)
    icmp_code = fluxo_bruto.get('IcmpCode', 0)
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
                print(f"[🚨 ALERTA DE INTRUSÃO] {obter_identidade_fluxo(metadados_fluxo)}")
                print(f"    Veredito: {classe.upper()} | Confiança: {confianca:.4f} | Ramo: {ramo}\n")
            elif classe == "Rejeitado":
                print(f"[⚠️ TRÁFEGO SUSPEITO] {obter_identidade_fluxo(metadados_fluxo)}")
                print(f"    Ação: FLUXO REJEITADO (Incerteza Operacional) | Confiança Máxima: {confianca:.4f}\n")
            else:
                # Tráfego benigno classificado de forma limpa
                print(f"[✅ Benigno] {obter_identidade_fluxo(metadados_fluxo)} -> Processado por: {ramo}")
        else:
            print(f"[-] Erro de comunicação externa: Resposta HTTP {resposta.status_code} recebida do servidor de IA.")
            
    except requests.exceptions.RequestException as e:
        print(f"[-] Falha na requisição: API inacessível ou tempo limite esgotado. Detalhes: {e}")

def obter_identidade_fluxo(fluxo):
    """Extrai informações de endereçamento da camada de rede (IP e Portas) para fins de log local."""
    src_ip = fluxo.get('SrcAddr', 'Desconhecido')
    dst_ip = fluxo.get('DstAddr', 'Desconhecido')
    src_port = fluxo.get('SrcPort', '?')
    dst_port = fluxo.get('DstPort', '?')
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