"""
Script para treinar um modelo de linguagem usando Hugging Face Transformers e Datasets.

Este exemplo carrega dados de perguntas e respostas de um arquivo JSON,
formata os dados, tokeniza, configura o treinamento e salva o modelo ajustado.

Requisitos:
- transformers
- datasets
- torch
"""

import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

def carregar_dados_json(caminho_arquivo):
    """
    Carrega dados de um arquivo JSON.
    Args:
        caminho_arquivo (str): Caminho para o arquivo JSON.
    Returns:
        list: Lista de exemplos do arquivo.
    """
    with open(caminho_arquivo, 'r') as f:
        return json.load(f)

def formatar_exemplo(exemplo):
    """
    Formata um exemplo para o modelo.
    Args:
        exemplo (dict): Exemplo contendo 'question' e 'answer'.
    Returns:
        dict: Exemplo formatado com chave 'text'.
    """
    return {"text": f"Pergunta: {exemplo['question']}\nResposta: {exemplo['answer']}"}

def tokenizar_exemplo(exemplos, tokenizer, max_length=128):
    """
    Tokeniza os exemplos usando o tokenizer do modelo.
    Args:
        exemplos (dict): Dicionário com chave 'text'.
        tokenizer: Tokenizer do modelo.
        max_length (int): Tamanho máximo dos tokens.
    Returns:
        dict: Exemplos tokenizados.
    """
    return tokenizer(exemplos["text"], truncation=True, max_length=max_length)

def main():
    # 1. Carregar dados
    dados_brutos = carregar_dados_json('train.json')

    # 2. Formatar dados
    dados_formatados = [formatar_exemplo(exemplo) for exemplo in dados_brutos]

    # 3. Converter para Dataset Hugging Face
    dataset = Dataset.from_list(dados_formatados)

    # 4. Carregar modelo e tokenizer pré-treinados
    modelo_nome = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    model = AutoModelForCausalLM.from_pretrained(modelo_nome)

    # 5. Ajustar token de padding se necessário
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # 6. Tokenizar o dataset
    def tokenize_fn(examples):
        return tokenizar_exemplo(examples, tokenizer)

    dataset_tokenizado = dataset.map(tokenize_fn, batched=True)

    # 7. Configurar argumentos de treinamento
    args_treinamento = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=200,
        push_to_hub=False,
    )

    # 8. Inicializar Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 9. Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=args_treinamento,
        train_dataset=dataset_tokenizado,
        data_collator=data_collator,
    )

    # 10. Treinar o modelo
    trainer.train()

    # 11. Salvar o modelo ajustado
    trainer.save_model("./fine_tuned_model")

if __name__ == "__main__":
    main()