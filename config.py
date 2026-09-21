# ============================
# Configurazione del server
# ============================

# URL del server compatibile con l'API OpenAI che ospita il modello
SERVER_URL = "http://localhost:8000/v1"

# API Key del server
API_KEY = "dummy"

# ============================
# Configurazione dei file
# ============================

# Percorso del file CSV di input
INPUT_CSV = "dati/prompt.csv"

# Percorso del file CSV di output
OUTPUT_CSV = "output/output.csv"

# Percorso del file CSV contenente le valutazioni del Judge
JUDGE_OUTPUT_CSV = "output/judge_output.csv"

# ============================
# Configurazione dei modelli
# ============================

# Nome dei modelli da utilizzare
MODELS = [
    "Qwen2.5-VL-7B-Instruct-Q4_K_M",
]

# Modello utilizzato come giudice
JUDGE_MODEL = "NOME_DEL_MODELLO"

# ============================
# Configurazione del batching
# ============================

BATCH_SIZE = 4

# ============================
# Parametri di generazione
# ============================

MAX_TOKENS = 300 # Numero massimo di token generati

DECODING_CONFIGS = {

    # Decodifica deterministica
    "deterministic": {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42
    },

    # Decodifica stocastica
    "stochastic": {
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42
    }
}

# ============================
# Parametri del Judge
# ============================

# Parametri del modello Judge
JUDGE_TEMPERATURE = 0.0

