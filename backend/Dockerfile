# 1. Usa uma imagem leve do Python
FROM python:3.10-slim

# 2. Define o diretório de trabalho dentro do container
WORKDIR /code

# 3. Copia o arquivo de dependências e instala as bibliotecas
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copia o resto dos arquivos (app.py e final_AllExit_v2.pth)
COPY . .

# 5. Comando para iniciar a API com o Uvicorn
# NOTA: O Hugging Face Spaces exige obrigatoriamente a porta 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]