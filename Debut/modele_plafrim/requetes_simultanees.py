from openai import AsyncOpenAI
import time
import json
import asyncio

# Crée un client OpenAI pointant vers le serveur vLLM local
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "./Qwen2.5-3B-Instruct"


async def envoyer_requete(id_prompt, prompt):
    start_time = time.time()
    final_answer = ""
    ttft = None

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],  # 'user' est souvent mieux pour tester
            stream=True,
            max_tokens=100,
            temperature=0,
        )

        async for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    if ttft is None:
                        ttft = time.time() - start_time
                        print(f"Req{id_prompt}, ttft{ttft}")

        duration = time.time() - start_time

    except Exception as e:
        print(f"Erreur pour la requête {id_prompt} : {e}")
        return 0, 0

    return duration, ttft


async def main():
    resultats = {}  # Dictionnaire pour stocker les dictionnaires de résultats
    nbr_requetes=[]
    for i in range(1,33):
        nbr_requetes.append(i)
    print(f"Nombre de requêtes simultanées à tester : {nbr_requetes}")
    tailles_tokens = [
        512,
        1024,
        2048,
        4096,
        16384,
        # 32768,
        # 47104,
        # 65536,
        # 98304,
        # 131072,
    ]  # Tailles de prompt à tester
    for n in nbr_requetes:
        print(f"\n--- Test avec {n} requêtes simultanées ---")
        resultats[f"{n} requêtes"] = []  # Initialisation de la liste pour ce nombre de requêtes
        for q in tailles_tokens:
            contenu_prompt = "lol " * (q-30)  # Génère un prompt de la taille souhaitée (en tokens) ici il faut enlevé 30 pour avoir le nombres de tokens requis
            taches = [envoyer_requete(i, contenu_prompt) for i in range(n)]
            retours = await asyncio.gather(*taches)
            total_duration = sum(r[0] for r in retours)
            total_ttft = sum(r[1] for r in retours)
            duree_moyenne = total_duration / n
            ttft_moyen = total_ttft / n
            resultats[f"{n} requêtes"].append(
                {
                    "taille_tokens": q,
                    "duree_moyenne": round(duree_moyenne, 4),
                    "ttft_moyen": round(ttft_moyen, 4),
                }
            )

    # Sauvegarde en JSON
    with open("resultats_bench.json", "w") as f:
        json.dump(resultats, f, indent=4)

    print("\nTests terminés ! Résultats sauvegardés.")
    return resultats


if __name__ == "__main__":
    final_data = asyncio.run(main())
    print(final_data)
