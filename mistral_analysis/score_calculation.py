import torch
from sentence_transformers import SentenceTransformer, util
from comet.models import download_model, load_from_checkpoint

gpu_av = torch.cuda.is_available()
meta_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def compute_meta_score(wikidata_entity_names, model_entities_predicted):
    meta_score_mentions = []
    if wikidata_entity_names and model_entities_predicted and len(model_entities_predicted) == len(wikidata_entity_names):
        for entity, model_entity in zip(wikidata_entity_names, model_entities_predicted):
            emb_entity = meta_model.encode(entity, convert_to_tensor=True)
            emb_model_entity = meta_model.encode(model_entity, convert_to_tensor=True)
            score = float(util.cos_sim(emb_entity, emb_model_entity).item())
            meta_score_mentions.append(score)
    return meta_score_mentions

def load_comet_model():
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)

def compute_comet_score(model, source, translation, targets):
    try:
        comet_data = [{"src": source, "mt": translation, "ref": t["translation"]} for t in targets]
        return max(model.predict(comet_data, batch_size=8, gpus=1 if gpu_av else 0)[0])
    except Exception as e:
        print(f"Error computing COMET score: {e}")
        return None