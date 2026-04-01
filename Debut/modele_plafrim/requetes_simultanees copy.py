from openai import OpenAI
import time
import json

# Crée un client OpenAI pointant vers le serveur vLLM local
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "./Qwen2.5-3B-Instruct"
prompt = "Hello"


def envoyer_requete(prompt):
    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],  # 'user' est souvent mieux pour tester
        stream=True,
        max_tokens=100,
        temperature=0,
    )

    final_answer = ""
    ttft = None

    for chunk in response:
        # On accède aux attributs avec des points . au lieu de [""]
        if chunk.choices:
            content = chunk.choices[0].delta.content

            # content peut être None ou une chaîne vide au tout début
            if content:
                if ttft is None:
                    ttft = time.time() - start_time

                final_answer += content
                # Optionnel : voir le texte s'afficher en temps réel
                print(content, end="", flush=True)

    duration = time.time() - start_time
    return final_answer, duration, ttft


final_answer, duration, ttft = envoyer_requete(prompt)
print(f"Final answer: {final_answer}")
print(f"Duration: {duration}")
print(f"TTFT: {ttft}")


