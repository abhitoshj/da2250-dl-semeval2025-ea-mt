from langchain.prompts import PromptTemplate

entity_extraction_prompt = PromptTemplate(
    input_variables=["texts"],
    template='''You are a named entity recognition (NER) expert.

For each of the following English sentences, extract all named entities (e.g., people, places, organizations, TV series, movies, books).

Instructions:
- ONLY extract named entities that appear EXACTLY and VERBATIM in the sentence.
- DO NOT return alternate names, inferred references, or canonical forms.
- DO NOT perform translation, rewriting, or guessing.
- DO NOT infer likely entities or use context to deduce names.
- An entity is valid ONLY if it is an exact substring match found in the sentence.
- If an entity is not present word-for-word in the sentence, DO NOT include it.
- DO NOT return partial entities or reformatted names.

Output format:
- Return a single JSON array.
- Each item must be an object with these two fields:
  - "Source": the original sentence.
  - "Entities": a list of string values. Each string must be copied directly from the sentence. Each must be an exact substring of the Source.

Rules:
- DO NOT include any reasoning or commentary.
- DO NOT include any Markdown formatting like triple backticks.
- The output MUST be valid JSON that can be parsed by Python's json.loads().

Texts:
{texts}
'''
)

entity_rethinking_prompt = PromptTemplate(
    input_variables=["sentence", "candidate"],
    template="""
You are an expert in Named Entity Recognition (NER). Named entities can be people, places, organizations, TV series, movies, books, etc.

Task:
Given a sentence and a candidate phrase, your goal is to identify the **most complete named entity** from the sentence that includes the candidate. This helps verify whether the candidate is a full entity, a partial one, or invalid.

Guidelines:
- If the candidate is a **subset of a longer named entity**, return the **full entity** from the sentence as-is.
- If the candidate **fully matches** a named entity in the sentence, return it.
- If the candidate is **not part of any valid named entity** in the sentence, return an empty list.
- Always return the named entity **verbatim**, exactly as it appears in the sentence (including casing, punctuation, etc.).
- Do not add inferred terms or modify the sentence.
- Named entities should not be a category or type. 

Response format:
Return a valid JSON object with the following keys:
- "sentence": the original input sentence.
- "entities": a list with either the corrected entity string or an empty list if not found.

Constraints:
- Output only the JSON. No markdown, no code blocks, no commentary.

Input:
Sentence: "{sentence}"
Candidate: "{candidate}"

Output:
"""
)

translation_prompt = PromptTemplate(
    input_variables=["sentence", "language", "entities"], # type: ignore
    template="""
You are a professional translator with expertise in high-fidelity, fluent translations that preserve named entities.
 
Translate the following English sentence into {language}. The translation MUST meet the following constraints:
 
1. The meaning is preserved **accurately** and the sentence reads naturally to native speakers.
2. All named entities from the list {entities} MUST appear in the translated sentence, translated correctly into {language}, matching real-world equivalents if they exist.
3. Use **natural phrasing and correct grammar** in {language}.
4. Avoid literal word-for-word translation and aim for native-like fluency.
5. Do NOT hallucinate or modify entity names. Only translate using the list provided.
6. Entity translation is mandatory unless it violates grammar or meaning in the {language}.
7. If an entity cannot be translated after all attempts, only then include it in its English form.
8. The output must be a valid JSON string that can be parsed by the Python json.loads() function.
9. Do not include any code fragments such as ``` or ```json in the output.
10. Do not include any additional text or explanations.
 
Format your response strictly as:
 
{{
  "translation": "<natural and accurate translated sentence>",
  "entities": ["<translated_entity1>", "<translated_entity2>", ...]
}}
 
Sentence: "{sentence}"
Entities: {entities}
"""
)