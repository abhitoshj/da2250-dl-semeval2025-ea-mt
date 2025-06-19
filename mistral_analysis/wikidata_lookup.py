import requests
import time

def find_best_match(entity_name, search_results):
    entity_name_lower = entity_name.lower()
    for result in search_results:
        label = result.get('label', '').lower()
        if entity_name_lower == label:
            return result
    for result in search_results:
        label = result.get('label', '').lower()
        if entity_name_lower in label:
            return result
    return search_results[0]

def fetch_wikidata_label(entity_id, target_locale):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            entity = resp.json()["entities"].get(entity_id, {})
            label = entity.get("labels", {}).get(target_locale, {}).get("value") \
                or entity.get("labels", {}).get("en", {}).get("value")
            if label:
                return label
    except Exception as e:
        print(f"Error fetching Wikidata for {entity_id}: {e}")
    return ''

def wikidata_translate_entityId(entity_name):
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': entity_name
    }
 
    try:
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"[WARN] Wikidata API failed for '{entity_name}' with status {response.status_code}")
            return None
        time.sleep(0.5)
 
        data = response.json()
        if not data.get('search'):
            return None

        qid = find_best_match(entity_name, data['search'])['id']
        if qid:
            return qid

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] HTTP error for '{entity_name}': {e}")
    except ValueError as e:
        print(f"[ERROR] JSON decode error for '{entity_name}': {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error for '{entity_name}': {e}")
 
    return None

def extract_entity_translation(entity_name, target_lang):
    qid = wikidata_translate_entityId(entity_name)
    english_label = fetch_wikidata_label(qid, "en")
    translated_label = fetch_wikidata_label(qid, target_lang)
    
    return {
        "qid": qid,
        "english": english_label,
        "translated": translated_label
    }