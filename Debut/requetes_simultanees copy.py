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

# for line in response.iter_lines():
#     if not line:
#         continue
#     if line.startswith(b"data: "):
#         json_str = line[6:].decode("utf-8").strip()
#         if json_str == "[DONE]":
#             break
#         try:
#             chunk = json.loads(json_str)
#             if "choices" in chunk and len(chunk["choices"]) > 0:
#                 content = chunk["choices"][0]["delta"].get("content", "")
#                 print(
#                     content, end="", flush=True
#                 )  # Affiche le contenu au fur et à mesure
#         except json.JSONDecodeError:
#             print(f"Erreur de décodage JSON : {json_str}")
#             continue

# # Affiche la réponse générée par le modèle
# print(response.choices[0].message.content)
