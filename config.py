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
INPUT_CSV = "dati/combinazioni_prompt_tirocinio.csv"

# Cartella principale contenente il dataset delle immagini.
IMAGE_DIR = "immagini"

# Percorso del file CSV di output
OUTPUT_CSV = "output/greedy.csv"

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

# Numero di richieste inviate contemporaneamente al server.
BATCH_SIZE = 4

# ============================
# Parametri di generazione
# ============================

# Numero massimo di token generati dal modello per ogni risposta
MAX_TOKENS = 300 

# Configurazioni di decoding utilizzate durante l'esperimento.
DECODING_CONFIGS = {

    # Decodifica deterministica
    "deterministic": {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42
    },
}

# ============================
# Parametri del Judge
# ============================

# Parametri di generazione del modello Judge.
JUDGE_TEMPERATURE = 0.0

