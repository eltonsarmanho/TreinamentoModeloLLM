from transformers import pipeline, AutoTokenizer

# Carregue o tokenizer do mesmo diretório do modelo treinado
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Crie o pipeline de geração de texto usando o modelo treinado
generator = pipeline("text-generation", model="./fine_tuned_model", tokenizer=tokenizer,truncation=True)

# Defina a nova pergunta que você quer fazer ao modelo
pergunta = "O que é um servidor público conforme a Lei Complementar nº 40/1992?"

# Formate a pergunta da mesma forma que você formatou os dados de treinamento
input_text = f"Pergunta: {pergunta}\nResposta:"

# Use o gerador para obter a resposta
response = generator(input_text, max_new_tokens=150, num_return_sequences=1)

# Imprima a resposta gerada
print(response[0]['generated_text'])