import json
import glob
import tqdm
import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from framework import get_language_name

model_name = "gemma3_1b"
llm = OllamaLLM(model="gemma3:1b")
""" prompt = PromptTemplate(
    input_variables=["text", "language"],
    template='''Translate the following sentence to {language}.
    
    Text: {text}
    The text may contain named entities, perform entity aware translation to {language}.
    Only output the translation in JSON format with the key "Translation".
    Escape the quotes in the JSON string so that it can be parsed by json.loads in Python. 
    Do not use [ or ] in the output. There should be no json arrays in the output.
    Do not include any code fragments such as ``` or ```json in the output.
    Example output: {{"Translation": "Bonjour le monde"}}
    Do not include any additional text or explanations.'''
) """
prompt = PromptTemplate(
    input_variables=["text", "language"],
    template='''Translate the following sentence to {language}.
    Text: {text}
    Only output the translated text.
    Do not include any additional text or explanations.'''
)

chain = prompt | llm

input_data_folder = "./data/semeval.validation.v2-889a1492ba6c3791baa8f4224bc8e685/validation"
jsonl_files = glob.glob(f"{input_data_folder}/*.jsonl")

output_prediction_dir = os.path.join("data/predictions", model_name, "validation")
os.makedirs(output_prediction_dir, exist_ok=True)

results = []
for file_path in jsonl_files:
    filename = os.path.basename(file_path)
    outfile_path = os.path.join(output_prediction_dir, filename)

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    results = []
    for idx, record in enumerate(data, 1):
        id = record['id']
        source = record['source']
        source_locale = record['source_locale']
        source_language = get_language_name(source_locale)
        target_locale = record['target_locale']
        target_language = get_language_name(target_locale)
        result = chain.invoke({"text": source, "language": target_language})
        """         
        result = result.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for source: {source}, result: {result}")
            continue 
        """
        model_translation = result.strip()
        results.append({
            "id": id,
            "source_language": source_language,
            "target_language": target_language,
            "text": source,
            "prediction": model_translation,
        })

        if 'pbar' not in locals():
            pbar = tqdm.tqdm(total=len(data))

        pbar.update(1)

        if idx % 10 == 0 or idx == len(data):
            with open(outfile_path, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Translations saved to {outfile_path}")

if 'pbar' in locals():
    pbar.close()

