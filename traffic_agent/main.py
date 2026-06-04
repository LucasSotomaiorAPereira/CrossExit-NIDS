from scapy.all import sniff

def processar_pacote(pacote):
    # Exibe o resumo do pacote
    print(pacote.show())

# sniff(filter="ip", prn=processar_pacote, store=0)
# filter: opcional (ex: "tcp", "udp", "port 80")
# prn: função a ser chamada para cada pacote capturado
# store: 0 evita guardar os pacotes na memória (importante para capturas longas)

print("Iniciando sniffing...")
sniff(prn=processar_pacote, store=0)