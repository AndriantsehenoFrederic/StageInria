from openai import OpenAI
import time
import json

# Crée un client OpenAI pointant vers le serveur vLLM local
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = "./Qwen2.5-3B-Instruct"
prompt = "lol" * 512


def envoyer_requete(prompt):
    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],  # 'user' est souvent mieux pour tester
        stream=False,
        max_tokens=20,
        temperature=0,
    )
    taille_prompt = response.usage.prompt_tokens
    return taille_prompt


final_answer = envoyer_requete(prompt)
print(final_answer)
