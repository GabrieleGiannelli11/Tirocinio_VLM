# Probing controfattuale di modelli Vision-Language: sensibilità a modificatori valutativi su immagini di occupazioni

Pipeline di inferenza sviluppata durante il tirocinio universitario per l'esecuzione di esperimenti controllati su Vision Language Models (VLM).

## Descrizione

Questo repository contiene la pipeline di inferenza sviluppata durante il tirocinio universitario relativo al progetto *"Probing controfattuale di modelli Vision-Language: sensibilità a modificatori valutativi su immagini di occupazioni"*.

L'obiettivo del progetto è studiare come le risposte di un Vision Language Model (VLM) cambino quando, a parità di immagine, viene modificata la formulazione del prompt attraverso l'utilizzo di modificatori valutativi (ad esempio "buon medico" e "cattivo medico").

La pipeline implementata in questo repository automatizza l'esecuzione di esperimenti controfattuali su Vision Language Models (VLM), combinando automaticamente un dataset di immagini di occupazioni con un insieme di condizioni di prompt definite in un file CSV.

Per ogni immagine, la pipeline applica tutte le combinazioni di system prompt e user prompt previste dall'esperimento, invia le richieste a un modello Vision Language compatibile con l'API OpenAI e salva le risposte generate insieme ai metadati dell'esperimento in un file CSV.

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
│   └── combinazioni_prompt_tirocinio.csv
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

- **dati/** contiene il file `combinazioni_prompt_tirocinio.csv`, che definisce tutte le condizioni sperimentali dell'esperimento.
- **immagini/** contiene il dataset di immagini organizzato in sottocartelle per professione. Le immagini non sono incluse nel repository e devono essere inserite dall'utente prima dell'esecuzione.
- **models/** contiene i file del modello Vision Language utilizzato dal server (se eseguito localmente). I file del modello non sono inclusi nel repository e devono essere scaricati separatamente.
- **output/** contiene i file generati automaticamente durante l'esecuzione dell'esperimento. La cartella è inizialmente vuota.

### Dataset di immagini

La cartella immagini/ contiene il dataset di immagini organizzato in sottocartelle corrispondenti alle professioni.
La pipeline ricerca automaticamente tutte le immagini presenti nella cartella immagini/ e nelle relative sottocartelle (.png, .jpg, .jpeg), senza che i percorsi delle immagini siano specificati nel file CSV.

Durante la preparazione di ogni richiesta viene ricavata automaticamente la professione associata all'immagine e il placeholder {occupation} presente nello user prompt viene sostituito con la relativa professione (ad esempio doctor, teacher o construction manager).
Questa sostituzione viene effettuata automaticamente prima dell'invio della richiesta al modello Vision Language.

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

- URL del server compatibile con OpenAI;
- uno o più modelli Vision Language da utilizzare;
- percorsi dei file di input e output;
- dimensione del batch (BATCH_SIZE);
- configurazioni di decoding (DECODING_CONFIGS).

## Modello Vision Language

Per motivi di dimensione, i file del modello Vision Language non sono inclusi in questo repository.

Per utilizzare la pipeline è necessario scaricare manualmente un modello compatibile con l'API OpenAI e inserirne i file nella cartella `models/`.

Durante lo sviluppo del progetto è stato utilizzato il modello **Qwen2.5-VL-7B-Instruct**, nei formati:

- `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- `Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf`

I file del modello sono disponibili su Hugging Face:

https://huggingface.co/DhruvalLabs/Qwen2.5-VL-7B-Instruct-GGUF

Una volta scaricato il modello, configurare il file `config.py` con il nome del modello e gli altri parametri necessari, quindi avviare il server compatibile con l'API OpenAI.

## Configurazioni di decoding

Le configurazioni di decoding sono definite nel file config.py tramite il dizionario DECODING_CONFIGS.
Ogni configurazione specifica i parametri di generazione (temperature, top_p e seed) utilizzati durante l'inferenza.

Per gli esperimenti iniziali viene utilizzata una configurazione deterministica (greedy):

temperature = 0.0
top_p = 1.0
seed = 42

La struttura del progetto permette di aggiungere facilmente ulteriori configurazioni di decoding (ad esempio stocastiche) senza modificare la pipeline.

## Esecuzione in batch

Per ridurre il tempo complessivo di esecuzione dell'esperimento, la pipeline utilizza il batching.
Le richieste vengono raggruppate in batch di dimensione configurabile (BATCH_SIZE) ed eseguite in parallelo tramite ThreadPoolExecutor.

Ogni batch contiene coppie immagine-prompt già preparate (immagine codificata in Base64, system prompt, user prompt e metadati sperimentali). Quando il batch raggiunge la dimensione configurata, viene elaborato dal modello e la pipeline crea automaticamente un nuovo batch per continuare l'esperimento.

Questa modalità consente di eseguire esperimenti su un numero elevato di immagini e condizioni di prompt mantenendo una pipeline scalabile e configurabile.

## Utilizzo

1. Inserire le immagini da analizzare nella cartella `immagini/`. 
2. Preparare il file `combinazioni_prompt_tirocinio.csv` nella cartella `dati/`, specificando per ogni immagine il relativo percorso e il prompt da utilizzare. Il valore della colonna image_path deve corrispondere al percorso dell'immagine all'interno della cartella immagini/.
3. Configurare il file `config.py` con il modello, l'URL del server, i percorsi dei file e i parametri di generazione.
4. Avviare il server compatibile con l'API OpenAI utilizzando il modello Vision Language desiderato.
5. Eseguire:

```bash
python main.py
```

## Output

Al termine dell'esecuzione vengono generati i seguenti file:

- greedy.csv: contiene tutte le risposte generate dal modello insieme ai metadati sperimentali (immagine, professione, combinazione di prompt, modello, decoding e risposta).
- experiment_config.json: configurazione completa dell'esperimento (modello, server, batch, decoding e percorsi utilizzati).
- experiment_metrics.json: contiene le metriche di ogni inferenza (numero di token e tempi di generazione) e le statistiche riassuntive dell'esperimento, incluso il tempo totale di esecuzione.
