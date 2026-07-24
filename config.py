# ============================
# Configurazione del progetto
# ============================

# URL del server compatibile con l'API OpenAI che ospita il modello
SERVER_URL = "http://localhost:8000/v1"

# Nome del modello da utilizzare
MODEL_NAME = "Qwen2.5-VL-7B-Instruct-Q4_K_M"

# Percorso del file CSV di input
INPUT_CSV = "dati/prompt.csv"

# Percorso del file CSV di output
OUTPUT_CSV = "output/output.csv"

# API Key del server
API_KEY = "dummy"

# ============================
# Parametri di generazione
# ============================

TEMPERATURE = 0.2 # Controlla la casualità della generazione
MAX_TOKENS = 300 # Numero massimo di token generati
TOP_P = 1.0 # Limita il campionamento ai token più probabili
SEED = 42 # Seed per rendere la generazione riproducibile
