import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. Definição Exata da Arquitetura do teu Artigo/Notebook
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

# Inicialização do FastAPI
app = FastAPI(title="Reliable Cross-Dataset Early Exits NIDS API")

# Modelo de Dados Pydantic para validação das requisições
class PredictionRequest(BaseModel):
    features: List[float]

# 2. Definição dos Limiares Duplos Otimizados via Grid Search (Exemplo do teu treino)
# Altere estes valores de acordo com os melhores resultados salvos em 'grid_search_history.csv'
T_ATK1  = 0.85   # Limiar de Ataque no Ramo 1
T_NORM1 = 0.85   # Limiar de tráfego Normal no Ramo 1
T_ATK2  = 0.85   # Limiar de Ataque no Ramo 2
T_NORM2 = 0.85   # Limiar de tráfego Normal no Ramo 2

# Carregamento Global do Modelo
device = torch.device("cpu")
model = IDSBranchyNet()

@app.on_event("startup")
def load_model():
    """Carrega os pesos do modelo PyTorch no arranque da API."""
    model_path = "aloha.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[-] Arquivo de pesos {model_path} não foi encontrado na raiz.")
    
    # Carrega mapeando para CPU para evitar quebra em servidores sem GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Pesos do IDSBranchyNet carregados com sucesso em modo de avaliação.")

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Validação do tamanho das características recebidas do traffic_agent.py
    if len(request.features) != INPUT_DIM:
        raise HTTPException(status_code=400, detail=f"O modelo espera exatamente {INPUT_DIM} características.")
    
    # Conversão para Tensor do PyTorch
    input_tensor = torch.tensor([request.features], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # --- PASSO 1: Executa a inferência leve do Ramo 1 (Early Exit)
        logits1 = model.forward_exit1(input_tensor)
        probs1 = F.softmax(logits1, dim=1)
        confianca1, pred_class1 = torch.max(probs1, dim=1)
        
        c1 = confianca1.item()
        p1 = pred_class1.item() # 0 = Normal, 1 = Ataque
        
        # Define o limiar dinâmico com base na predição do Ramo 1
        limiar1 = T_ATK1 if p1 == 1 else T_NORM1
        
        # Condição de Saída Antecipada (Early Exit)
        if c1 > limiar1:
            classe_str = "Ataque" if p1 == 1 else "Normal"
            return {
                "classe": classe_str,
                "ramo_saida": "Ramo 1 (Early Exit)",
                "confianca": c1
            }
        
        # --- PASSO 2: Se a confiança for baixa, ativa o Ramo 2 (Rede Profunda)
        logits2 = model.forward_exit2(input_tensor)
        probs2 = F.softmax(logits2, dim=1)
        confianca2, pred_class2 = torch.max(probs2, dim=1)
        
        c2 = confianca2.item()
        p2 = pred_class2.item()
        
        # Define o limiar dinâmico para o Ramo 2
        limiar2 = T_ATK2 if p2 == 1 else T_NORM2
        
        # Condição de Decisão vs Rejeição por incerteza
        if c2 > limiar2:
            classe_str = "Ataque" if p2 == 1 else "Normal"
            return {
                "classe": classe_str,
                "ramo_saida": "Ramo Final (Complexo)",
                "confianca": c2
            }
        else:
            # Ativa o mecanismo de dupla rejeição defendido no artigo
            return {
                "classe": "Rejeitado",
                "ramo_saida": "Nenhum (Margem de Incerteza)",
                "confianca": c2
            }

@app.get("/")
def home():
    return {"status": "Online", "projeto": "Reliable Cross-Dataset NIDS via Early-Exit Branching"}