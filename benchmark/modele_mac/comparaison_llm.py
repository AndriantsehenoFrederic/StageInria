import requests
import json
import time
from dotenv import load_dotenv
import os

load_dotenv()

qwen_vl_32b_instruct_mlx_prompts = {
    "prompt512": "lol" * 503,
    "prompt1k": "lol" * 1015,
    "prompt2k": "lol" * 2039,
    "prompt4k": "lol" * 4087,
    "prompt16k": "lol" * 16375,
    "prompt32k": "lol" * 32759,
    "prompt48k": "lol" * 47095,
    "prompt64k": "lol" * 65527,
    "prompt96k": "lol" * 98295,
    "prompt128k": "lol" * 131063,
}

# Configuration
URL = "https://unpalpablely-vibronic-leonore.ngrok-free.dev/api/v1/chat/completions"
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "qwen3-vl-32b-instruct-mlx"  # Ne pas oublier de changer le modèle ici


def envoyer_requete(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        # Désactive le streaming pour avoir une réponse d'un bloc
        "stream": True,
        "max_tokens": 100,
    }

    full_answer = ""
    first_time_token = None

    try:
        start_time = time.time()
        response = requests.post(
            URL, headers=headers, data=json.dumps(data), stream=True, timeout=3600
        )
        response.raise_for_status()  # Vérifie si la requête a réussi (code 200)
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                json_str = line[6:].decode("utf-8").strip()
                if json_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        # print(chunk)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content and first_time_token is None:
                            first_time_token = time.time() - start_time
                            first_time_token_time = first_time_token

                        full_answer += content

                except json.JSONDecodeError:
                    print(f"Erreur de décodage JSON : {json_str}")
                    continue

        end_time = time.time()
        duration = end_time - start_time
        return full_answer, duration, first_time_token_time

    except requests.exceptions.RequestException as e:
        error_msg = f"Erreur de connexion : {e}"
        if "response" in locals() and response is not None:
            error_msg += f" | Contenu : {response.text}"
        return error_msg, 0, 0
    except KeyError:
        return "Erreur : Format de réponse inattendu."


# Test
if __name__ == "__main__":
    resultat = {}

# On crée une entrée pour le modèle actuel si elle n'existe pas
if MODEL_NAME not in resultat:
    resultat[MODEL_NAME] = []

for (
    cle,
    valeur,
) in (
    qwen_vl_32b_instruct_mlx_prompts.items()
):  # changer le nom du dictionnaire en fonction du modèle qu'on veut testé
    reponse, duration, first_time_token_time = envoyer_requete(prompt=valeur)

    resultat[MODEL_NAME].append(
        {
            "nb_tokens_attendus": cle[6:],
            "duree_totale": duration,
            "ttft": first_time_token_time,
            "reponse_courte": reponse[:50] + "...",
        }
    )
print(resultat)


import csv

nom_propre = MODEL_NAME.replace("/", "_").replace(":", "_")
nom_fichier_csv = f"resultats_{nom_propre}.csv"

fieldnames = [
    "Nombre de tokens",
    "Durée totale de réponse",
    "Temps pour recevoir le premier token",
]

# Ouvrir le fichier en mode append ('a') et écrire l'en-tête seulement si le fichier est nouveau
with open(nom_fichier_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Écrire les lignes des résultats
    for i in resultat[MODEL_NAME]:
        writer.writerow(
            {
                "Nombre de tokens": i["nb_tokens_attendus"],
                "Durée totale de réponse": i["duree_totale"],
                "Temps pour recevoir le premier token": i["ttft"],
            }
        )
    print("fichier suavegardé!!!")


import matplotlib.pyplot as plt
import pandas as pd

nom_fichier_csv = "resultats_qwen_qwen3-coder-30b.csv"  # changer le nom du fichier csv en fonction du modèle testé
df = pd.read_csv(nom_fichier_csv)

plt.figure(figsize=(10, 6))
plt.title("{MODEL_NAME} - Durée totale de réponse en fonction du nombre de tokens")
plt.plot(
    df["Nombre de tokens"],
    df["Durée totale de réponse"],
    marker="o",
    label="qwen/qwen3-coder-30b",
    color="blue",
)
