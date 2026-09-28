# ============================
# Import delle librerie
# ============================

import base64                # Per convertire le immagini in Base64
import json                  # Per lavorare con dati JSON
import os                    # Per lavorare con file e percorsi
import time                  # Per misurare il tempo di esecuzione dell'esperimento
from concurrent.futures import ThreadPoolExecutor, as_completed # Per eseguire le richieste della batch in parallelo

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
# Funzione per trovare le immagini
# ============================

def get_image_paths(image_folder):
    """
    Restituisce la lista dei percorsi di tutte le immagini
    presenti nella cartella principale e in tutte le sue
    sottocartelle.

    Il dataset è organizzato in categorie di professioni e,
    per ogni professione, nelle cartelle NEGATIVE, NEUTRAL
    e POSITIVE.

    Args:
        image_folder: cartella principale contenente le immagini.

    Returns:
        Lista dei percorsi delle immagini supportate.
    """

    image_paths = []

    for root, _, files in os.walk(image_folder):

        for file in files:

            extension = os.path.splitext(file)[1].lower()

            if extension in [".png", ".jpg", ".jpeg"]:

                image_paths.append(
                    os.path.join(root, file)
                )

    image_paths.sort()

    return image_paths

# ============================
# Funzione per ottenere l'occupation dell'immagine
# ============================

def get_occupation(image_path):
    """
    Restituisce la professione (occupation) associata
    all'immagine leggendo il nome della cartella della
    professione nel percorso dell'immagine.

    Args:
        image_path: percorso del file immagine.

    Returns:
        L'occupation associata all'immagine.
    """

    occupation_folder = os.path.basename(
        os.path.dirname(image_path)
    )

    occupation_map = {

        # Professioni con stereotipi particolarmente evidenti
        "Aircraft pilots": "aircraft pilot",
        "Barbers": "barber",
        "Childcare workers": "childcare worker",
        "Dressmakers": "dressmaker",
        "Executive secretaries": "executive secretary",
        "Flight attendants": "flight attendant",
        "Hairdressers, hairstylists and cosmetologists": "hairdresser, hairstylist or cosmetologist",
        "Laundry workers": "laundry worker",
        "Maids": "maid",
        "Manicurists and pedicurists": "manicurist or pedicurist",
        "Massage therapists": "massage therapist",
        "Personal care aides": "personal care aide",
        "Police officers": "police officer",
        "Receptionists and information clerks": "receptionist or information clerk",
        "Registered nurses": "registered nurse",

        # Professioni di cura, servizio e basso status
        "Bakers": "baker",
        "Dishwashers": "dishwasher",
        "Elementary school teachers": "elementary school teacher",
        "Healthcare social workers": "healthcare social worker",
        "Home health aides": "home health aide",
        "Kindergarten teachers": "kindergarten teacher",
        "Phlebotomists": "phlebotomist",
        "Preschool teachers": "preschool teacher",
        "Special education teachers": "special education teacher",

        # Professioni di responsabilità e prestigio
        "Chefs": "chef",
        "Chief executives": "chief executive",
        "Construction managers": "construction manager",
        "Dentists": "dentist",
        "Legislators": "legislator",
        "Producers and directors": "producer or director",
        "Sales managers": "sales manager",
        "Surgeons": "surgeon",
        "Veterinarians": "veterinarian",

        # Professioni manuali, di forza e sicurezza
        "Blockmasons": "blockmason",
        "Carpenters": "carpenter",
        "Construction laborers": "construction laborer",
        "Electricians": "electrician",
        "Farming, fishing and forestry occupations": "worker in farming, fishing or forestry",
        "Firefighters": "firefighter",
        "Plumbers": "plumber",
        "Security guards": "security guard",

        # Professioni STEM
        "Aerospace engineers": "aerospace engineer",
        "Civil engineers": "civil engineer",
        "Computer and information systems managers": "computer and information systems manager",
        "Computer hardware engineers": "computer hardware engineer",
        "Computer programmers": "computer programmer",
        "Electrical engineers": "electrical engineer",
        "Information security analysts": "information security analyst",
        "Mechanical engineers": "mechanical engineer",
        "Web developers": "web developer"
    }

    if occupation_folder not in occupation_map:
        raise ValueError(
            f"Occupation non riconosciuta per l'immagine: {image_path}"
        )

    return occupation_map[occupation_folder]
    
# ============================
# Funzione per chiedere al modello una descrizione dell'immagine
# ============================

def generate_description(
    client, 
    model_name, 
    image_b64, 
    mime_type, 
    system_prompt,
    user_prompt, 
    temperature, 
    max_tokens, 
    top_p, 
    seed
):
    """
    Invia un'immagine, un system prompt e un user prompt al modello VLM.

    Args:
        client: client OpenAI compatibile.
        model_name: nome del modello da utilizzare.
        image_b64: immagine codificata in Base64.
        mime_type: tipo MIME dell'immagine.
        system_prompt: istruzione iniziale che definisce il comportamento del modello. Se è vuoto, il modello utilizza il comportamento di default.
        user_prompt: testo da inviare al modello.
        temperature: controlla la casualità della generazione.
        max_tokens: numero massimo di token della risposta.
        top_p: parametro di campionamento del modello.
        seed: seed del generatore casuale per rendere l'esperimento riproducibile.

    Returns:
        La risposta generata dal modello.
    """

    # Costruisce la lista dei messaggi da inviare al modello.
    messages = []

    # Aggiunge il system prompt solo se esiste
    if pd.notna(system_prompt) and str(system_prompt).strip():
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Messaggio dell'utente (prompt + immagine)
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}"
                }
            }
        ]
    })

    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
        messages=messages
    )

    return response

# ============================
# Funzione per elaborare un batch di richieste
# ============================

def generate_batch(
    client,
    batch,
    model_name,
    temperature,
    max_tokens,
    top_p,
    seed
):
    """
    Invia contemporaneamente le richieste contenute nel batch.

    Args:
        client: client OpenAI compatibile.
        batch: lista di richieste già preparate contenente:
        - immagine codificata in Base64;
        - system prompt (quando presente);
        - user prompt già preparato, con eventuale sostituzione
        del placeholder {occupation};
        - metadati della combinazione di prompt
        system prompt e user prompt.
        model_name: nome del modello da utilizzare.
        temperature: temperatura di generazione.
        max_tokens: numero massimo di token della risposta.
        top_p: parametro di campionamento.
        seed: seed per la riproducibilità.

    Returns:
        Lista dei risultati, nello stesso ordine degli elementi
        presenti nel batch.
    """

    batch_responses = [None] * len(batch)

    with ThreadPoolExecutor(
        max_workers=config.BATCH_SIZE
    ) as executor:

        futures = {}

        for index, item in enumerate(batch):

            future = executor.submit(
                generate_description,
                client,
                model_name,
                item["image_b64"],
                item["mime_type"],
                item["system_prompt"],
                item["user_prompt"],
                temperature,
                max_tokens,
                top_p,
                seed
            )

            futures[future] = index

        for future in as_completed(futures):
                
            index = futures[future]

            try:

                batch_responses[index] = {
                    "success": True,
                    "response": future.result()
                }

            except Exception as e:

                batch_responses[index] = {
                    "success": False,
                    "error": str(e)
                }

    return batch_responses

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
# Lettura del file delle combinazioni di prompt
# ============================

# Il CSV contiene tutte le combinazioni di system prompt
# e user prompt generate per l'esperimento.
# Le immagini vengono recuperate automaticamente dalla
# cartella del dataset.

prompt_df = pd.read_csv(
    config.INPUT_CSV,
    sep=";"
)

total_prompt_conditions = len(prompt_df)

# Recupera i percorsi di tutte le immagini
image_paths = get_image_paths(config.IMAGE_DIR)

total_images = len(image_paths)

# Numero totale di coppie immagine-condizione
total_input_pairs = (
    total_images * total_prompt_conditions
)

results = [] # Lista che conterrà i risultati di tutte le inferenze

metrics = [] # Lista che conterrà le metriche di tutte le inferenze

experiment_start = time.time()

# ============================
# Elaborazione dei batch
# ============================

for batch_start in range(0, total_input_pairs, config.BATCH_SIZE):

    batch_end = min(
        batch_start + config.BATCH_SIZE,
        total_input_pairs
    )

    batch = []

    for task_index in range(batch_start, batch_end):

        # Individua l'immagine e la condizione di prompt
        image_index = task_index // total_prompt_conditions
        
        prompt_index = task_index % total_prompt_conditions

        image_path = image_paths[image_index]

        row = prompt_df.iloc[prompt_index]

        # Controlla che il file immagine esista
        if not os.path.exists(image_path):

            # Crea un risultato di errore per ogni modello e configurazione di decoding
            for model_name in config.MODELS:

                # Crea un risultato di errore per ogni configurazione di decoding
                for decoding_name, decoding_config in config.DECODING_CONFIGS.items():

                    results.append({
                        "image": image_path,
                        "combination_id": row["combination_id"],
                        "system_prompt_id": row["system_prompt_id"],
                        "dimension": row["dimension"],
                        "system_condition": row["system_condition"],
                        "target_group": row["target_group"],
                        "orientation": row["orientation"],
                        "system_language": row["system_language"],
                        "system_prompt": row["system_prompt"],
                        "prompt_id": row["prompt_id"],
                        "prompt_condition": row["prompt_condition"],
                        "paraphrase": row["paraphrase"],
                        "prompt_language": row["prompt_language"],
                        "user_prompt": row["user_prompt"],
                        "model": model_name,
                        "decoding": decoding_name,
                        "temperature": decoding_config["temperature"],
                        "top_p": decoding_config["top_p"],
                        "seed": decoding_config["seed"],
                        "response": "ERRORE: Immagine non trovata"
                    })

            continue

        # Ottiene l'occupation associata all'immagine
        occupation = get_occupation(image_path)
        
        # Prepara il user prompt
        user_prompt = row["user_prompt"]
        
        # Sostituisce {occupation} solo nei prompt che lo contengono
        if pd.notna(user_prompt):
            user_prompt = str(user_prompt).replace(
                "{occupation}",
                occupation
            )
        else:
            user_prompt = ""

        # Codifica l'immagine in Base64
        image_b64 = encode_image(image_path)

        # Ottiene il MIME type dell'immagine
        mime_type = get_mime_type(image_path)

        # Controlla che il formato sia supportato
        if mime_type is None:

            # Esegue l'inferenza con ogni modello
            for model_name in config.MODELS:

                # Crea un risultato di errore per ogni configurazione di decoding
                for decoding_name, decoding_config in config.DECODING_CONFIGS.items():

                    results.append({
                        "image": image_path,
                        "combination_id": row["combination_id"],
                        "system_prompt_id": row["system_prompt_id"],
                        "dimension": row["dimension"],
                        "system_condition": row["system_condition"],
                        "target_group": row["target_group"],
                        "orientation": row["orientation"],
                        "system_language": row["system_language"],
                        "system_prompt": row["system_prompt"],
                        "prompt_id": row["prompt_id"],
                        "prompt_condition": row["prompt_condition"],
                        "paraphrase": row["paraphrase"],
                        "prompt_language": row["prompt_language"],
                        "user_prompt": row["user_prompt"],
                        "model": model_name,
                        "decoding": decoding_name,
                        "temperature": decoding_config["temperature"],
                        "top_p": decoding_config["top_p"],
                        "seed": decoding_config["seed"],
                        "response": "ERRORE: Formato immagine non supportato"
                    })

            continue

        # Aggiunge alla batch una richiesta completa contenente:
        # immagine, prompt preparati, occupation e metadati della
        # combinazione di prompt.
        batch.append({
            "image_path": image_path,
            "combination_id": row["combination_id"],
            "system_prompt_id": row["system_prompt_id"],
            "dimension": row["dimension"],
            "system_condition": row["system_condition"],
            "target_group": row["target_group"],
            "orientation": row["orientation"],
            "system_language": row["system_language"],
            "system_prompt": row["system_prompt"],
            "prompt_id": row["prompt_id"],
            "prompt_condition": row["prompt_condition"],
            "paraphrase": row["paraphrase"],
            "prompt_language": row["prompt_language"],
            "user_prompt": user_prompt,
            "occupation": occupation,
            "image_b64": image_b64,
            "mime_type": mime_type
        })

    if not batch:
        continue

    # Esegue l'inferenza con ogni modello
    for model_name in config.MODELS:

        # Esegue l'inferenza con ogni configurazione di decoding
        for decoding_name, decoding_config in config.DECODING_CONFIGS.items():

            try:

                batch_results = generate_batch(
                    client,
                    batch,
                    model_name,
                    decoding_config["temperature"],
                    config.MAX_TOKENS,
                    decoding_config["top_p"],
                    decoding_config["seed"]
                )

            except Exception as e:

                # Salva l'errore relativo alla batch corrente
                for item in batch:

                    results.append({
                        "image": item["image_path"],
                        "combination_id": item["combination_id"],
                        "system_prompt_id": item["system_prompt_id"],
                        "dimension": item["dimension"],
                        "system_condition": item["system_condition"],
                        "target_group": item["target_group"],
                        "orientation": item["orientation"],
                        "system_language": item["system_language"],
                        "system_prompt": item["system_prompt"],
                        "prompt_id": item["prompt_id"],
                        "prompt_condition": item["prompt_condition"],
                        "paraphrase": item["paraphrase"],
                        "prompt_language": item["prompt_language"],
                        "user_prompt": item["user_prompt"],
                        "occupation": item["occupation"],
                        "model": model_name,
                        "decoding": decoding_name,
                        "temperature": decoding_config["temperature"],
                        "top_p": decoding_config["top_p"],
                        "seed": decoding_config["seed"],
                        "response": f"ERRORE: {e}"
                    })

                continue

            # Associa ogni risposta alla rispettiva coppia
            for item, batch_result in zip(batch, batch_results):

                # Controlla se la richiesta è andata a buon fine
                if not batch_result["success"]:

                    results.append({
                        "image": item["image_path"],
                        "combination_id": item["combination_id"],
                        "system_prompt_id": item["system_prompt_id"],
                        "dimension": item["dimension"],
                        "system_condition": item["system_condition"],
                        "target_group": item["target_group"],
                        "orientation": item["orientation"],
                        "system_language": item["system_language"],
                        "system_prompt": item["system_prompt"],
                        "prompt_id": item["prompt_id"],
                        "prompt_condition": item["prompt_condition"],
                        "paraphrase": item["paraphrase"],
                        "prompt_language": item["prompt_language"],
                        "user_prompt": item["user_prompt"],
                        "occupation": item["occupation"],
                        "model": model_name,
                        "decoding": decoding_name,
                        "temperature": decoding_config["temperature"],
                        "top_p": decoding_config["top_p"],
                        "seed": decoding_config["seed"],
                        "response": f"ERRORE: {batch_result['error']}"
                    })

                    continue

                try:

                    # Recupera la vera risposta del server
                    response = batch_result["response"]

                    # Estrae le metriche del backend
                    backend_metrics = extract_backend_metrics(response)

                    # Estrae il testo della risposta generata dal modello
                    description = response.choices[0].message.content

                    # Salva il risultato dell'inferenza
                    result = {
                        "image": item["image_path"],
                        "combination_id": item["combination_id"],
                        "system_prompt_id": item["system_prompt_id"],
                        "dimension": item["dimension"],
                        "system_condition": item["system_condition"],
                        "target_group": item["target_group"],
                        "orientation": item["orientation"],
                        "system_language": item["system_language"],
                        "system_prompt": item["system_prompt"],
                        "prompt_id": item["prompt_id"],
                        "prompt_condition": item["prompt_condition"],
                        "paraphrase": item["paraphrase"],
                        "prompt_language": item["prompt_language"],
                        "user_prompt": item["user_prompt"],
                        "occupation": item["occupation"],
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
                        "image": item["image_path"],
                        "combination_id": item["combination_id"],
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

                    # Salva l'errore relativo alla singola coppia
                    results.append({
                        "image": item["image_path"],
                        "combination_id": item["combination_id"],
                        "system_prompt_id": item["system_prompt_id"],
                        "dimension": item["dimension"],
                        "system_condition": item["system_condition"],
                        "target_group": item["target_group"],
                        "orientation": item["orientation"],
                        "system_language": item["system_language"],
                        "system_prompt": item["system_prompt"],
                        "prompt_id": item["prompt_id"],
                        "prompt_condition": item["prompt_condition"],
                        "paraphrase": item["paraphrase"],
                        "prompt_language": item["prompt_language"],
                        "user_prompt": item["user_prompt"],
                        "occupation": item["occupation"],
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

average_time_per_image = round(
    experiment_time / len(image_paths),
    2
)

if metrics:

    summary = {
        "total_images": total_images,
        "total_prompt_conditions": total_prompt_conditions,
        "total_input_pairs": total_input_pairs,
        "total_expected_inferences": (
            total_input_pairs
            * len(config.MODELS)
            * len(config.DECODING_CONFIGS)
        ),
        "successful_inferences": len(metrics), 
        "total_experiment_time_seconds": experiment_time,
        "average_time_per_image_seconds": average_time_per_image,

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
        "total_images": total_images,
        "total_prompt_conditions": total_prompt_conditions,
        "total_input_pairs": total_input_pairs,
        "total_expected_inferences": (
            total_input_pairs
            * len(config.MODELS)
            * len(config.DECODING_CONFIGS)
        ),
        "successful_inferences": 0,
        "total_experiment_time_seconds": experiment_time,
        "average_time_per_image_seconds": average_time_per_image
    }

save_experiment_metrics(
    summary,
    metrics,
    config.OUTPUT_CSV
)