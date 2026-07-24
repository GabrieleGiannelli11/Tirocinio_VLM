# Probing controfattuale di modelli Vision-Language: sensibilità a modificatori valutativi su immagini di occupazioni

Pipeline di inferenza sviluppata durante il tirocinio universitario per l'esecuzione di esperimenti controllati su Vision Language Models (VLM).

## Descrizione

Questo repository contiene la pipeline di inferenza sviluppata durante il tirocinio universitario relativo al progetto *"Probing controfattuale di modelli Vision-Language: sensibilità a modificatori valutativi su immagini di occupazioni"*.

L'obiettivo del progetto è studiare come le risposte di un Vision Language Model (VLM) cambino quando, a parità di immagine, viene modificata la formulazione del prompt attraverso l'utilizzo di modificatori valutativi (ad esempio "buon medico" e "cattivo medico").

La pipeline implementata in questo repository automatizza l'esecuzione degli esperimenti, inviando immagini e prompt a un modello Vision Language compatibile con l'API OpenAI e salvando le risposte generate in un file CSV.

Attualmente la pipeline permette di:

- leggere un dataset di immagini e prompt da un file CSV;
- verificare la presenza e il formato delle immagini;
- convertire automaticamente le immagini in Base64;
- inviare richieste ad un Vision Language Model tramite un server compatibile con l'API OpenAI;
- configurare i principali parametri di generazione tramite un file di configurazione;
- salvare le risposte generate in un nuovo file CSV;
- gestire automaticamente gli errori durante l'inferenza;
- salvare automaticamente la configurazione dell'esperimento in formato JSON;
- salvare le metriche di inferenza (token e tempi di generazione) e un riepilogo statistico dell'esperimento in formato JSON.

## Struttura del progetto

```text
.
├── dati/
│   └── prompt.csv
├── immagini/
├── models/
├── output/
├── .gitignore
├── config.py
├── main.py
├── README.md
└── requirements.txt
```

### Cartelle

- **dati/** contiene il file `prompt.csv`, che specifica il percorso delle immagini e il prompt associato a ciascuna di esse.
- **immagini/** contiene le immagini da analizzare. Le immagini non sono incluse nel repository e devono essere inserite dall'utente prima dell'esecuzione.
- **models/** contiene i file del modello Vision Language utilizzato dal server (se eseguito localmente). I file del modello non sono inclusi nel repository e devono essere scaricati separatamente.
- **output/** contiene i file generati automaticamente durante l'esecuzione dell'esperimento. La cartella è inizialmente vuota.

### File principali

- **main.py** implementa l'intera pipeline di inferenza.
- **config.py** contiene tutti i parametri configurabili del progetto (modello, URL del server, percorsi dei file e parametri di generazione).
- **requirements.txt** contiene l'elenco delle dipendenze Python necessarie per eseguire il progetto.

## Requisiti

- Python 3.13
- pandas
- openai Python SDK
- llama.cpp server (oppure un server compatibile con l'API OpenAI)
- modello Vision Language (ad esempio Qwen2.5-VL)

## Installazione

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

Configurare successivamente il file `config.py` con:

- URL del server;
- nome del modello;
- percorsi dei file di input e output;
- parametri di generazione.

## Modello Vision Language

Per motivi di dimensione, i file del modello Vision Language non sono inclusi in questo repository.

Per utilizzare la pipeline è necessario scaricare manualmente un modello compatibile con l'API OpenAI e inserirne i file nella cartella `models/`.

Durante lo sviluppo del progetto è stato utilizzato il modello **Qwen2.5-VL-7B-Instruct**, nei formati:

- `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- `Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf`

I file del modello sono disponibili su Hugging Face:

https://huggingface.co/DhruvalLabs/Qwen2.5-VL-7B-Instruct-GGUF

Una volta scaricato il modello, configurare il file `config.py` con il nome del modello e gli altri parametri necessari, quindi avviare il server compatibile con l'API OpenAI.

## Utilizzo

1. Inserire le immagini da analizzare nella cartella `immagini/`. 
2. Preparare il file `prompt.csv` nella cartella `dati/`, specificando per ogni immagine il relativo percorso e il prompt da utilizzare. Il valore della colonna image_path deve corrispondere al percorso dell'immagine all'interno della cartella immagini/.
3. Configurare il file `config.py` con il modello, l'URL del server, i percorsi dei file e i parametri di generazione.
4. Avviare il server compatibile con l'API OpenAI utilizzando il modello Vision Language desiderato.
5. Eseguire:

```bash
python main.py
```

## Output

Al termine dell'esecuzione vengono generati i seguenti file:

- output.csv: contiene le descrizioni generate dal modello.
- experiment_config.json: contiene i parametri utilizzati durante l'esperimento.
- experiment_metrics.json: contiene le metriche di ogni inferenza (numero di token e tempi di generazione) e le statistiche riassuntive dell'esperimento, incluso il tempo totale di esecuzione.
