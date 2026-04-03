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
    resultats_bruts = []
    nbr_requetes = [1, 2, 4, 8, 16, 32]
    tailles_tokens = [512, 1024, 2048, 4096, 16384]
    nb_repetitions = 10

    for n in nbr_requetes:
        print(f"Test avec {n} requêtes simultanées...")

        for q in tailles_tokens:
            print(f"  Tokens: {q}")

            for essai in range(nb_repetitions):
                contenu_prompt = "lol " * (
                    q - 30
                )  # on enlève 30 pour avoir pile la bonne taille de tokens, peut etre changer en fonction du modèle utilisé
                taches = [envoyer_requete(i, contenu_prompt) for i in range(n)]
                retours = await asyncio.gather(*taches)
                for r in retours:
                    resultats_bruts.append(
                        {
                            "nb_requetes_simultanees": n,
                            "taille_tokens": q,
                            "essai_index": essai,
                            "duree": r[0],
                            "ttft": r[1],
                        }
                    )

    print(f"\nTests terminés ! {len(resultats_bruts)} mesures collectées.")
    return resultats_bruts


if __name__ == "__main__":
    final_data = asyncio.run(main())
    print(final_data)
    with open(f"resultats_requetes_simultanees.json", "w") as f:
        json.dump(final_data, f, indent=4)
