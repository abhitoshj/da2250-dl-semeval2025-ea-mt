import json
import glob
import tqdm
import demjson3
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain

from entity_extraction import (
    extract_capitalized_phrases,
    extract_after_prepositions,
    extract_quoted_entities,
    extract_hyphenated_entities,
    extract_entities_with_numbers_or_roman,
    validate_entities
)
from wikidata_lookup import extract_entity_translation, fetch_wikidata_label
from prompt_templates_zero_prompt import (
    entity_extraction_prompt,
    entity_rethinking_prompt,
    translation_prompt
)
from score_calculation import compute_meta_score, load_comet_model, compute_comet_score


def load_all_jsonl_files(folder_path):
    data = []
    for file_path in glob.glob(f"{folder_path}/**/*.jsonl", recursive=True):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    return data


def get_language_name(short_code):
    lang_map = {
        'ar': 'Arabic', 'zh': 'Chinese (Traditional)', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'es': 'Spanish',
        'th': 'Thai', 'tr': 'Turkish', 'en': 'English'
    }
    return lang_map.get(short_code, short_code)


def run_pipeline():
    jsonl_folder = "Validation/data/references/validation"
    all_jsonl_data = load_all_jsonl_files(jsonl_folder)
    print(f"Loaded {len(all_jsonl_data)} records from {jsonl_folder}")

    # Use Ollama only in main
    llm = OllamaLLM(model="mistral")
    chain_extract = LLMChain(llm=llm, prompt=entity_extraction_prompt)
    chain_rethink = LLMChain(llm=llm, prompt=entity_rethinking_prompt)
    chain_translate = LLMChain(llm=llm, prompt=translation_prompt)

    comet_model = load_comet_model()
    output_file = "outputs/translated_mistral_zero_prompt.jsonl"
    results = []

    pbar = tqdm.tqdm(total=len(all_jsonl_data))

    for record in all_jsonl_data:
        source = record['source']
        language = get_language_name(record['target_locale'])
        wikidata_ids = [record['wikidata_id']]

        # Extract named entities
        try:
            raw_entities = chain_extract.invoke({"texts": source})
            entity_data = json.loads(raw_entities)
        except Exception:
            try:
                entity_data = demjson3.decode(raw_entities)
            except Exception as e2:
                print(f"Failed to recover batch with demjson3: {e2}")
                continue

        local_entities = set(
            extract_capitalized_phrases(source) +
            extract_after_prepositions(source) +
            extract_quoted_entities(source) +
            extract_hyphenated_entities(source) +
            extract_entities_with_numbers_or_roman(source)
        )

        cleaned_entity_list = []
        if isinstance(entity_data, dict):
            cleaned_entity_list.extend(validate_entities(entity_data.get('Entities', []), source))
        elif isinstance(entity_data, list):
            for item in entity_data:
                cleaned_entity_list.extend(validate_entities(item.get('Entities', []), source))

        # Rethink entities
        for entity in local_entities:
            if entity not in cleaned_entity_list:
                correction = chain_rethink.invoke({"sentence": source, "candidate": entity})
                
                try:
                    new_data = json.loads(correction)
                except json.JSONDecodeError as e:
                    try:
                        new_data = demjson3.decode(correction)
                    except Exception as e2:
                        print(f"Failed to recover batch with demjson3: {e2}")
                        continue
                
                if new_data.get('entities'):
                    cleaned_entity_list.extend(new_data['entities'])
                    cleaned_entity_list.append(entity)

        cleaned_entity_list = list(set([x.strip() for x in cleaned_entity_list if x.strip()]))

        # Remove duplicate entries
        duplicate_entities = []
        for i in range(len(cleaned_entity_list)):
            for j in range(len(cleaned_entity_list)):
                if i != j and cleaned_entity_list[i] in cleaned_entity_list[j]:
                    duplicate_entities.append(cleaned_entity_list[i])

        final_entity_list = []
        for ent in cleaned_entity_list:
            if ent not in duplicate_entities:
                final_entity_list.append(ent)

        # Translate named entities using Wikidata
        model_entities = []
        for item in final_entity_list:
            ent = extract_entity_translation(item, record['target_locale'])
            if ent['qid']:
                model_entities.append(ent['english'])

        wikidata_entity_names = [fetch_wikidata_label(qid, record['target_locale']) for qid in wikidata_ids]

        # Translate sentence with constraint
        try:
            raw_translated = chain_translate.invoke({
                "sentence": source,
                "language": language,
                "entities": ", ".join(model_entities)
            })
            raw_translated = json.loads(raw_translated)
        except Exception:
            raw_translated = demjson3.decode(raw_translated)

        # Scores
        comet_score = compute_comet_score(comet_model, source, raw_translated['translation'], record["targets"])
        meta_score_mentions = compute_meta_score(wikidata_entity_names, raw_translated['entities'])

        results.append({
            "source": source,
            "target_locale": record['target_locale'],
            "model_translation": raw_translated['translation'],
            "entities": wikidata_entity_names,
            "model_entities": raw_translated['entities'],
            "meta_score_entities": meta_score_mentions,
            "comet_score": comet_score
        })

        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        pbar.update(1)

    pbar.close()
    print(f"Translations saved to {output_file}*")


if __name__ == "__main__":
    run_pipeline()