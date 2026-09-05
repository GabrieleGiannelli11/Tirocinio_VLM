# ============================
# Import delle librerie
# ============================

import base64                # Per convertire le immagini in Base64
import json                  # Per lavorare con dati JSON
import os                    # Per lavorare con file e percorsi
import time                  # Per misurare il tempo di esecuzione dell'esperimento

import pandas as pd          # Per leggere e scrivere file CSV
from openai import OpenAI    # Client per comunicare con il server del modello AI

import config                # Importa le configurazioni definite in config.py

# ============================
# Funzione per codificare un'immagine in Base64
# ============================

def encode_image(image_path):
    """
    Converte un'immagine in una stringa Base64.

    Args:
        image_path: percorso del file immagine.

    Returns:
        La stringa Base64 dell'immagine.
    """
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
# ============================
# Funzione per ottenere il MIME type
# ============================

def get_mime_type(image_path):
    """
    Restituisce il MIME type di un'immagine.

    Args:
        image_path: percorso del file immagine.

    Returns:
        "image/png" oppure "image/jpeg" se il formato è supportato.
        None se il formato non è supportato.
    """

    extension = os.path.splitext(image_path)[1].lower()

    if extension == ".png":
        return "image/png"

    elif extension in [".jpg", ".jpeg"]:
        return "image/jpeg"

    return None
    
# ============================
# Funzione per chiedere al modello una descrizione dell'immagine
# ============================

def generate_description(
    client, 
    model_name, 
    image_b64, 
    mime_type, 
    prompt, 
    temperature, 
    max_tokens, 
    top_p, 
    seed
):
    """
    Invia un'immagine e un prompt al modello VLM.

    Args:
        client: client OpenAI compatibile.
        model_name: nome del modello da utilizzare.
        image_b64: immagine codificata in Base64.
        mime_type: tipo MIME dell'immagine.
        prompt: testo da inviare al modello.
        temperature: controlla la casualità della generazione.
        max_tokens: numero massimo di token della risposta.
        top_p: parametro di campionamento del modello.
        seed: seed del generatore casuale per rendere l'esperimento riproducibile.

    Returns:
        La risposta generata dal modello.
    """

    response = client.chat.completions.create(

        model=model_name,

        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        }
                    }

                ]
            }
        ],
    )

    return response

# ============================
# Funzione per estrarre le metriche del backend
# ============================

def extract_backend_metrics(response):
    """
    Estrae le metriche di esecuzione indipendentemente
    dal backend utilizzato.

    Args:
        response: risposta restituita dal server OpenAI compatibile.

    Returns:
        Un dizionario contenente:
        - prompt_ms
        - predicted_ms
        - predicted_per_second
    """

    # Caso llama.cpp
    if hasattr(response, "timings"):
        return response.timings

    # Caso vLLM
    if hasattr(response, "metrics") and response.metrics is not None:
        return {
            "prompt_ms": response.metrics["time_to_first_token_ms"],
            "predicted_ms": response.metrics["generation_time_ms"],
            "predicted_per_second": response.metrics["tokens_per_second"]
        }

    raise RuntimeError("Il backend non restituisce metriche.")

# ============================
# Funzione per salvare la configurazione dell'esperimento
# ============================

def save_experiment_config(
    models,
    max_tokens,
    decoding_configs,
    input_csv,
    output_csv
):
    """
    Salva i parametri utilizzati durante l'esperimento in un file JSON.

    Args:
        models: lista dei nomi dei modelli utilizzati.
        max_tokens: numero massimo di token.
        decoding_configs: configurazioni di decoding utilizzate.
        input_csv: percorso del file CSV di input.
        output_csv: percorso del file CSV di output.
    """

    experiment_config = {
        "models": models,
        "max_tokens": max_tokens,
        "decoding_configs": decoding_configs,
        "input_csv": input_csv
    }

    output_folder = os.path.dirname(output_csv)

    output_path = os.path.join(
        output_folder,
        "experiment_config.json"
    )

    with open(output_path, "w") as output_file:
        json.dump(experiment_config, output_file, indent=4)

# ============================
# Funzione per salvare le metriche dell'esperimento
# ============================

def save_experiment_metrics(
    summary,
    metrics,
    output_csv
):
    """
    Salva le metriche dell'esperimento in un file JSON.

    Args:
        summary: dizionario contenente le statistiche riassuntive dell'esperimento.
        metrics: lista contenente le metriche di ogni inferenza.
        output_csv: percorso del file CSV di output.   
    """

    output_folder = os.path.dirname(output_csv)

    output_path = os.path.join(
        output_folder,
        "experiment_metrics.json"
    )

    experiment_metrics = {
        "summary": summary,
        "metrics": metrics
    }

    with open(output_path, "w") as output_file:
        json.dump(experiment_metrics, output_file, indent=4)

# ============================
# Inizializzazione dell'esperimento
# ============================

client = OpenAI(
    base_url=config.SERVER_URL,
    api_key=config.API_KEY
)

save_experiment_config(
    config.MODELS,
    config.MAX_TOKENS,
    config.DECODING_CONFIGS,
    config.INPUT_CSV,
    config.OUTPUT_CSV
)

# ============================
# Lettura del file CSV
# ============================

# Il CSV contiene:
# - il percorso dell'immagine
# - il prompt da inviare al modello

df = pd.read_csv(config.INPUT_CSV, sep=";")

total_input_images = len(df) # Numero totale di immagini nel CSV

results = [] # Lista che conterrà i risultati di tutte le inferenze

metrics = [] # Lista che conterrà le metriche di tutte le inferenze

experiment_start = time.time()

# ============================
# Elaborazione di ogni riga del CSV
# ============================

for index, row in df.iterrows():

    # Controlla che il file immagine esista
    if not os.path.exists(row["image_path"]):

        # Crea un risultato di errore per ogni modello e configurazione di decoding
        for model_name in config.MODELS:

            # Crea un risultato di errore per ogni configurazione di decoding
            for decoding_name, decoding_config in config.DECODING_CONFIGS.items():

                results.append({
                    "image": row["image_path"],
                    "prompt": row["prompt"],
                    "model": model_name,
                    "decoding": decoding_name,
                    "temperature": decoding_config["temperature"],
                    "top_p": decoding_config["top_p"],
                    "seed": decoding_config["seed"],
                    "response": "ERRORE: Immagine non trovata"
                })

            continue

    image_b64 = encode_image(row["image_path"])

    mime_type = get_mime_type(row["image_path"])

    # Controlla che il formato sia supportato
    if mime_type is None:

        # Esegue l'inferenza con ogni modello
        for model_name in config.MODELS:

            # Crea un risultato di errore per ogni configurazione di decoding
            for decoding_name, decoding_config in config.DECODING_CONFIGS.items():

                results.append({
                    "image": row["image_path"],
                    "prompt": row["prompt"],
                    "model": model_name,
                    "decoding": decoding_name,
                    "temperature": decoding_config["temperature"],
                    "top_p": decoding_config["top_p"],
                    "seed": decoding_config["seed"],
                    "response": "ERRORE: Formato immagine non supportato"
                })

        continue

    # Esegue l'inferenza con ogni modello
    for model_name in config.MODELS:

        # Esegue l'inferenza con ogni configurazione di decoding
        for decoding_name, decoding_config in config.DECODING_CONFIGS.items():

            try:

                response = generate_description(
                    client,
                    model_name,
                    image_b64,
                    mime_type,
                    row["prompt"],
                    decoding_config["temperature"],
                    config.MAX_TOKENS,
                    decoding_config["top_p"],
                    decoding_config["seed"]
                )

                backend_metrics = extract_backend_metrics(response)

                # Estrae il testo della risposta generata dal modello
                description = response.choices[0].message.content

                # Salva il risultato dell'inferenza
                result = {
                    "image": row["image_path"],
                    "prompt": row["prompt"],
                    "model": model_name,
                    "decoding": decoding_name,
                    "temperature": decoding_config["temperature"],
                    "top_p": decoding_config["top_p"],
                    "seed": decoding_config["seed"],
                    "response": description
                }

                results.append(result)

                # Salva le metriche dell'inferenza corrente
                metric = {
                    "image": row["image_path"],
                    "model": model_name,
                    "decoding": decoding_name,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "prompt_ms": backend_metrics["prompt_ms"],
                    "predicted_ms": backend_metrics["predicted_ms"],
                    "predicted_per_second": backend_metrics["predicted_per_second"]
                }

                metrics.append(metric)

            except Exception as e:

                # Salva l'errore relativo a questa configurazione
                results.append({
                    "image": row["image_path"],
                    "prompt": row["prompt"],
                    "model": model_name,
                    "decoding": decoding_name,
                    "temperature": decoding_config["temperature"],
                    "top_p": decoding_config["top_p"],
                    "seed": decoding_config["seed"],
                    "response": f"ERRORE: {e}"
                })

# ============================
# Salvataggio dei risultati
# ============================

results_df = pd.DataFrame(results)

results_df.to_csv(
    config.OUTPUT_CSV,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

experiment_end = time.time()

experiment_time = round(
    experiment_end - experiment_start,
    2
)

# ============================
# Creazione del riepilogo delle metriche
# ============================

if metrics:

    summary = {
        "total_input_images": total_input_images,
        "successful_inferences": len(metrics), 
        "total_experiment_time_seconds": experiment_time,

        "average_prompt_tokens": round(
            sum(
                metric["prompt_tokens"]
                for metric in metrics
            ) / len(metrics),
            2
        ),

        "average_completion_tokens": round(
            sum(
                metric["completion_tokens"]
                for metric in metrics
            ) / len(metrics),
            2
        ),

        "average_total_tokens": round(
            sum(
                metric["total_tokens"]
                for metric in metrics
            ) / len(metrics),
            2
        ),

        "average_prompt_ms": round(
            sum(
                metric["prompt_ms"]
                for metric in metrics
            ) / len(metrics),
            2
        ),

        "average_predicted_ms": round(
            sum(
                metric["predicted_ms"]
                for metric in metrics
            ) / len(metrics),
            2
        ),

        "average_predicted_per_second": round(
            sum(
                metric["predicted_per_second"]
                for metric in metrics
            ) / len(metrics),
            2
        )
    }

else:

    summary = {
        "total_input_images": total_input_images,
        "successful_inferences": 0,
        "total_experiment_time_seconds": experiment_time
    }

save_experiment_metrics(
    summary,
    metrics,
    config.OUTPUT_CSV
)