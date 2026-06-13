import os
import sys
import glob
import shutil
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuração do logging para monitoramento
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 1. Parâmetros e Configuração do nProbe

NPROBE_FEATURES = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"
]

def build_nprobe_template(features):
    return "".join([f"%{feat}" for feat in features])

def run_nprobe_batch(pcap_path, temp_dump_dir, separator="#"):
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"Arquivo PCAP não encontrado no caminho fornecido: {pcap_path}")

    os.makedirs(temp_dump_dir, exist_ok=True)
    template_str = build_nprobe_template(NPROBE_FEATURES)
    
    comando = [
        "nprobe",
        "-i", pcap_path,
        "-V", "9",
        "--dont-reforge-time",
        "-T", template_str,
        "--dump-path", temp_dump_dir,
        "--dump-format", "t",
        "--csv-separator", separator
    ]
    
    result = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else "Erro desconhecido na execução do nProbe."
        raise RuntimeError(f"Falha na execução do binário nProbe: {error_msg}")

def consolidate_extracted_flows(temp_dump_dir, output_csv_path, separator="#"):
    arquivos_dump = glob.glob(os.path.join(temp_dump_dir, "**", "*"), recursive=True)
    arquivos_dump = [f for f in arquivos_dump if os.path.isfile(f)]
    if not arquivos_dump:
        logging.warning("Nenhum fluxo exportado foi gerado pelo nProbe. O arquivo PCAP pode estar vazio.")
        return pd.DataFrame()
        
    lista_dataframes = []
    
    for arquivo in arquivos_dump:
        try:
            df_temp = pd.read_csv(
                arquivo, sep=separator, names=NPROBE_FEATURES, header=0, comment="#", low_memory=False
            )
            if not df_temp.empty:
                lista_dataframes.append(df_temp)
        except Exception as e:
            logging.error(f"Erro ao analisar o arquivo de dump {arquivo}: {e}")
            
    if not lista_dataframes:
        logging.error("Nenhum fluxo válido pôde ser recuperado.")
        return pd.DataFrame()
        
    df_consolidado = pd.concat(lista_dataframes, ignore_index=True)
    df_consolidado.fillna(0, inplace=True)
    if output_csv_path is not None:
        df_consolidado.to_csv(output_csv_path, index=False)
    
    return df_consolidado

def split_pcap(pcap_file, output_dir, packets_per_file=2000):
    """Divide o arquivo PCAP em chunks menores via editcap."""
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "chunk.pcap")

    comando = [
        "editcap",
        "-c", str(packets_per_file),
        pcap_file,
        output_pattern,
    ]

    logging.info(f"Dividindo PCAP em partes de {packets_per_file} pacotes cada...")
    result = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else "Erro desconhecido na execução do editcap."
        raise RuntimeError(f"Falha ao dividir o PCAP com editcap: {error_msg}")

    chunks = sorted(glob.glob(os.path.join(output_dir, "chunk*.pcap")))
    logging.info(f"PCAP dividido em {len(chunks)} parte(s).")
    return chunks

def process_single_chunk(idx, chunk_path, temp_base_dir, total_chunks):
    logging.info(f"Processando parte {idx + 1}/{total_chunks}")
    temp_dump_dir = os.path.join(temp_base_dir, f"dump_{idx}")
    run_nprobe_batch(chunk_path, temp_dump_dir)

    df_chunk = consolidate_extracted_flows(temp_dump_dir, output_csv_path=None)
    return df_chunk

def extract_pcap_to_netflow(pcap_file, output_csv):
    diretorio_base = os.path.dirname(os.path.abspath(pcap_file))
    temp_base_dir = os.path.join(diretorio_base, ".temp_pcap_nprobe")
    temp_split_dir = os.path.join(temp_base_dir, "splits")
    try:
        chunks = split_pcap(pcap_file, temp_split_dir)

        lista_dataframes = []
        num_workers = min(os.cpu_count() or 1, len(chunks))
        logging.info(f"Iniciando ThreadPoolExecutor com {num_workers} workers para processar {len(chunks)} partes...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_single_chunk, idx, chunk_path, temp_base_dir, len(chunks))
                for idx, chunk_path in enumerate(chunks)
            ]
            for idx, future in enumerate(futures):
                try:
                    df_chunk = future.result()
                    if df_chunk is not None and not df_chunk.empty:
                        lista_dataframes.append(df_chunk)
                except Exception as e:
                    logging.error(f"Erro ao processar a parte {idx + 1}: {e}")

        if not lista_dataframes:
            logging.error("Nenhum fluxo válido extraído de nenhuma parte do PCAP.")
            return pd.DataFrame()

        df_final = pd.concat(lista_dataframes, ignore_index=True)
        df_final.fillna(0, inplace=True)
        df_final.to_csv(output_csv, index=False)
        logging.info(f"Total de {len(df_final)} fluxos consolidados de {len(chunks)} parte(s).")
        return df_final
    finally:
        if os.path.exists(temp_base_dir):
            logging.info("Limpando arquivos temporários do nProbe...")
            shutil.rmtree(temp_base_dir)


# 2. Definição da Arquitetura e Inferência do Modelo

INPUT_DIM = 32

class IDSBranchyNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=2):
        super(IDSBranchyNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.exit1_layers = nn.Sequential(
            nn.Linear(input_dim * 2, num_classes)
        )
        
        self.exit2_layers = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward_exit1(self, x):
        features = self.shared_layers(x)
        return self.exit1_layers(features)

    def forward_exit2(self, x):
        features = self.shared_layers(x)
        return self.exit2_layers(features)

def inferencia(model, x, device, params=None):
    model.eval()
    model.to(device)
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    
    if x.ndim == 1:
        x = x.unsqueeze(0)
        
    x = x.to(device)
    
    with torch.no_grad():
        logits1 = model.forward_exit1(x)
        probs1 = F.softmax(logits1, dim=1)
        confs1, preds1 = torch.max(probs1, dim=1)
        
        logits2 = model.forward_exit2(x)
        probs2 = F.softmax(logits2, dim=1)
        confs2, preds2 = torch.max(probs2, dim=1)
        
        if params is None:
            return {
                'exit1': {
                    'preds': preds1.cpu().numpy(),
                    'confs': confs1.cpu().numpy()
                },
                'exit2': {
                    'preds': preds2.cpu().numpy(),
                    'confs': confs2.cpu().numpy()
                }
            }
        
        t_atk1, t_norm1, t_atk2, t_norm2 = params
        
        thresh_tensor1 = torch.where(preds1 == 1, t_atk1, t_norm1).to(device)
        mask_exit1 = confs1 > thresh_tensor1
        
        thresh_tensor2 = torch.where(preds2 == 1, t_atk2, t_norm2).to(device)
        mask_exit2 = (~mask_exit1) & (confs2 > thresh_tensor2)
        
        mask_rejected = (~mask_exit1) & (~mask_exit2)
        
        final_preds = preds2.clone()
        final_preds[mask_exit1] = preds1[mask_exit1]
        
        final_confs = confs2.clone()
        final_confs[mask_exit1] = confs1[mask_exit1]
        
        exits_utilizados = torch.zeros_like(final_preds)
        exits_utilizados[mask_exit1] = 1  
        exits_utilizados[mask_exit2] = 2  
        exits_utilizados[mask_rejected] = -1
        
        return {
            'predicoes': final_preds.cpu().numpy(),
            'confiancas': final_confs.cpu().numpy(),
            'rota_saida': exits_utilizados.cpu().numpy()
        }


# 3. Orquestração e Execução do Pipeline

def main():
    if len(sys.argv) < 3:
        print("Uso correto: python3 nids_pipeline.py <arquivo_entrada.pcap> <arquivo_saida.csv>")
        sys.exit(1)
        
    pcap_input = sys.argv[1]
    csv_output = sys.argv[2]
    
    # Passo 1: Extração NetFlow via nProbe
    df_flows = extract_pcap_to_netflow(pcap_input, csv_output)
    
    if df_flows is None or df_flows.empty:
        logging.error("Nenhum fluxo extraído do PCAP. Pipeline abortado.")
        sys.exit(1)
        
    # Passo 2: Seleção e ordenação das features para o modelo
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
    
    df_features = df_flows[CARACTERISTICAS_MODELO]
    
    # Passo 3: Carregamento do modelo e do scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "model.pth")
    scaler_path = os.path.join(script_dir, "models", "minmax_scaler.pkl")

    model = IDSBranchyNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Pesos do modelo carregados com sucesso.")
    else:
        logging.error(f"Erro: O arquivo de pesos não foi encontrado em: {model_path}")
        sys.exit(1)
    model.to(device)
    
    if os.path.exists(scaler_path):
        loaded_scaler = joblib.load(scaler_path)
        logging.info("Normalizador carregado com sucesso.")
    else:
        logging.error(f"Erro: O arquivo do scaler não foi encontrado em: {scaler_path}")
        sys.exit(1)

    # Passo 4: Pré-processamento e inferência
    X = df_features.values

    X_df = pd.DataFrame(X, columns=CARACTERISTICAS_MODELO)

    if np.isinf(X_df.values).any():
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_df.fillna(0, inplace=True)

    X_scaled = loaded_scaler.transform(X_df.values)
    X_scaled = np.clip(X_scaled, 0, 1)

    melhores_params = (0.85, 0.85, 0.85, 0.85)
    
    logging.info(f"Executando inferência em lote para {len(X_scaled)} fluxos extraídos...")
    resultado = inferencia(
        model=model, 
        x=X_scaled,
        device=device, 
        params=melhores_params
    )

    # Passo 5: Atualização do relatório com predições
    df_flows['PREDICAO_CLASSE'] = resultado['predicoes']
    df_flows['GRAU_CONFIANCA'] = resultado['confiancas']
    df_flows['ROTA_SAIDA_EXIT'] = resultado['rota_saida']
    
    df_flows.to_csv(csv_output, index=False)
    logging.info(f"Pipeline concluído com sucesso! Resultados salvos em: {csv_output}")

    # Impressão das métricas de classificação
    preds = resultado['predicoes']
    exits = resultado['rota_saida']
    
    rejeitados = np.sum(exits == -1)
    benignos_saida1 = np.sum((preds == 0) & (exits == 1))
    benignos_saida2 = np.sum((preds == 0) & (exits == 2))
    ataques_saida1 = np.sum((preds == 1) & (exits == 1))
    ataques_saida2 = np.sum((preds == 1) & (exits == 2))

    logging.info("Métricas de Classificação")
    logging.info(f"Fluxos rejeitados: {rejeitados}")
    logging.info(f"Fluxos benignos na Saída 1: {benignos_saida1}")
    logging.info(f"Fluxos benignos na Saída 2: {benignos_saida2}")
    logging.info(f"Ataques na Saída 1: {ataques_saida1}")
    logging.info(f"Ataques na Saída 2: {ataques_saida2}")

if __name__ == "__main__":
    main()