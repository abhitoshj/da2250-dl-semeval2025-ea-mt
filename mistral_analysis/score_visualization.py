import json
from collections import defaultdict
import matplotlib.pyplot as plt
 
file_path = "outputs/translated_zero_prompt.jsonl"
 
# Load and process Mistral output
wikidata_scores = defaultdict(lambda: {"comet": [], "meta": []})
 
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        lang = item.get("target_locale", "unknown")
        if isinstance(item.get("comet_score"), (int, float)):
            wikidata_scores[lang]["comet"].append(item["comet_score"])
        metas = item.get("meta_score_entities", [])
        if isinstance(metas, list) and metas:
            wikidata_scores[lang]["meta"].append(sum(metas) / len(metas))
 
# Compute averages
languages = sorted(wikidata_scores.keys())
avg_comet = [sum(wikidata_scores[lang]["comet"]) / len(wikidata_scores[lang]["comet"]) for lang in languages]
avg_meta = [sum(wikidata_scores[lang]["meta"]) / len(wikidata_scores[lang]["meta"]) for lang in languages]
 
# Plot
x = range(len(languages))
width = 0.35
 
plt.figure(figsize=(12, 6))
plt.bar([i - width / 2 for i in x], avg_comet, width=width, label="COMET Score", color="#1f77b4")
plt.bar([i + width / 2 for i in x], avg_meta, width=width, label="M-ETA Score", color="#ff7f0e")
 
# Add annotations
for i, (c, m) in enumerate(zip(avg_comet, avg_meta)):
    plt.text(i - width / 2, c + 0.005, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
    plt.text(i + width / 2, m + 0.005, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
 
plt.xticks(x, languages)
plt.xlabel("Language")
plt.ylabel("Score")
plt.title("Mistral Model: Average COMET & M-ETA Scores by Language")
plt.legend()
plt.tight_layout()
 
# Save image
plt.savefig("outputs/zero_prompt.png")
plt.close()