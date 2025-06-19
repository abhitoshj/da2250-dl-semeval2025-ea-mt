import re

def extract_capitalized_phrases(text):
    return re.findall(r'\b(?:[A-Z][a-z]*\s?)+', text)

def extract_after_prepositions(text):
    return re.findall(r'\b(?:in|of|from|at|by|to|on)\s+([A-Z][a-zA-Z\'’\-]*(?:\s+[A-Z][a-zA-Z\'’\-]*)*)', text)

def extract_quoted_entities(text):
    matches = re.findall(r'"(.*?)"|“(.*?)”|\'(.*?)\'', text)
    return [group for tup in matches for group in tup if group]

def extract_hyphenated_entities(text):
    return re.findall(r'\b(?:[A-Z][a-z]+-)+[A-Z][a-z]+\b', text)

def extract_entities_with_numbers_or_roman(text):
    return re.findall(r'\b(?:[A-Z][a-z]+\s)*(?:[A-Z][a-z]+)\s(?:[IVXLCDM]+|\d+)\b', text)

def validate_entities(entities, source):
    return [entity for entity in entities if entity in source]