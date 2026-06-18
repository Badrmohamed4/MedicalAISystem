#!/usr/bin/env python3
"""
PubMed Knowledge Base Builder for MedicalAISystem
====================================================
Queries PubMed's free E-utilities API for peer-reviewed abstracts
across the 3 system domains (brain, lung, skin), cleans them,
and outputs an expanded medical_documents.json.

Run this from your project root:
    python build_pubmed_kb.py

Requires: requests (already in requirements.txt)
"""

import requests
import json
import time
import re
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUTPUT_PATH = "medical_ai_project/systems/medical_documents.json"

# Search queries per domain — each will pull multiple abstracts
SEARCH_QUERIES = {
    "brain": [
        "glioma symptoms treatment prognosis",
        "meningioma management surgery observation",
        "pituitary adenoma hormone vision",
        "brain tumor headache red flag",
        "brain tumor seizure first aid management",
        "brain tumor cognitive changes MRI findings",
        "glioma grade 3 4 prognosis survival",
        "brain tumor post surgery recovery complications",
        "brain tumor radiotherapy side effects",
        "FLAIR hyperintensity brain tumor meaning",
        "brain MRI T1 T2 enhancement tumor",
    ],
    "lung": [
        "lung adenocarcinoma symptoms staging treatment",
        "large cell lung carcinoma prognosis chemotherapy",
        "squamous cell lung carcinoma smoking central",
        "lung cancer cough hemoptysis red flag",
        "lung cancer shortness of breath management",
        "lung cancer chest pain differential",
        "lung cancer stage 3 4 survival rates",
        "lung cancer post lobectomy recovery",
        "lung cancer immunotherapy side effects",
        "lung nodule PET scan SUV meaning",
        "ground glass opacity lung cancer",
        "lung CT scan findings interpretation",
    ],
    "skin": [
        "melanoma ABCDE criteria biopsy staging",
        "melanoma sentinel lymph node prognosis",
        "psoriasis plaque guttate treatment biologics",
        "eczema atopic dermatitis flare triggers",
        "acne vulgaris cystic treatment isotretinoin",
        "melanoma itching bleeding warning signs",
        "skin lesion asymmetry border color change",
        "psoriasis psoriatic arthritis joint pain",
        "melanoma wide excision recovery scar",
        "melanoma stage 1 2 3 survival",
        "psoriasis phototherapy UVB side effects",
        "melanoma Breslow thickness Clark level",
        "skin biopsy pathology report terms",
    ],
    "emergency": [
        "medical emergency warning signs when to go ER",
        "high risk patient emergency protocol",
    ],
}

MAX_RESULTS_PER_QUERY = 5

# ── PUBMED API FUNCTIONS ──────────────────────────────────────────────────────

def search_pubmed(query, max_results=5):
    """Search PubMed and return list of PMIDs."""
    url = f"{PUBMED_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  ⚠️ Search failed for '{query}': {e}")
        return []

def fetch_abstracts(pmids):
    """Fetch abstracts for given PMIDs."""
    if not pmids:
        return []
    url = f"{PUBMED_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return parse_pubmed_xml(resp.text)
    except Exception as e:
        print(f"  ⚠️ Fetch failed for PMIDs {pmids}: {e}")
        return []

def parse_pubmed_xml(xml_text):
    """Parse PubMed XML and extract title + abstract text."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""
        abstract_elems = article.findall(".//AbstractText")
        abstract_parts = []
        for elem in abstract_elems:
            text = elem.text or ""
            label = elem.get("Label", "")
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)
        if abstract and len(abstract) > 50:
            articles.append({"title": title.strip(), "abstract": abstract.strip()})
    return articles

def clean_text(text):
    """Clean and normalize abstract text for the KB."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.strip()
    if len(text) > 1500:
        text = text[:1500].rsplit(" ", 1)[0] + "."
    return text

def build_document(title, abstract, domain, query_topic):
    """Build a document entry matching the existing JSON format."""
    full_text = f"{title}. {abstract}" if title else abstract
    full_text = clean_text(full_text)
    tags = [domain]
    topic_lower = query_topic.lower()
    condition_tags = {
        "glioma": ["glioma", "brain tumor"],
        "meningioma": ["meningioma", "brain tumor"],
        "pituitary": ["pituitary", "brain tumor"],
        "adenocarcinoma": ["adenocarcinoma", "lung cancer"],
        "large cell": ["large cell carcinoma", "lung cancer"],
        "squamous": ["squamous cell carcinoma", "lung cancer"],
        "melanoma": ["melanoma", "skin cancer"],
        "psoriasis": ["psoriasis"],
        "eczema": ["eczema", "atopic dermatitis"],
        "acne": ["acne"],
        "emergency": ["emergency", "high risk"],
        "seizure": ["seizure", "brain"],
        "headache": ["headache", "brain"],
        "cough": ["cough", "lung"],
        "MRI": ["MRI", "brain"],
        "CT": ["CT scan", "lung"],
        "biopsy": ["biopsy", "skin"],
        "staging": ["staging", "prognosis"],
        "recovery": ["post-surgery", "recovery"],
        "side effects": ["side effects", "treatment"],
    }
    for keyword, extra_tags in condition_tags.items():
        if keyword in topic_lower:
            tags.extend(extra_tags)
            break
    tags = list(dict.fromkeys(tags))
    return {"text": full_text, "tags": tags, "source": f"PUBMED_{domain.upper()}"}

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PubMed Knowledge Base Builder")
    print("  MedicalAISystem — Expanding RAG Documents")
    print("=" * 60)
    all_documents = []
    total_queries = sum(len(queries) for queries in SEARCH_QUERIES.values())
    processed = 0
    for domain, queries in SEARCH_QUERIES.items():
        print(f"\n📂 Domain: {domain.upper()}")
        for query in queries:
            processed += 1
            print(f"  [{processed}/{total_queries}] Searching: {query[:50]}...")
            pmids = search_pubmed(query, max_results=MAX_RESULTS_PER_QUERY)
            if not pmids:
                print(f"    → No results found")
                continue
            print(f"    → Found {len(pmids)} PMIDs, fetching abstracts...")
            articles = fetch_abstracts(pmids)
            for article in articles:
                doc = build_document(article["title"], article["abstract"], domain, query)
                all_documents.append(doc)
                print(f"    ✅ Added: {doc['text'][:60]}...")
            time.sleep(0.4)
    print(f"\n{"=" * 60}")
    print(f"  TOTAL DOCUMENTS COLLECTED: {len(all_documents)}")
    print(f"{"=" * 60}")
    if len(all_documents) == 0:
        print("\n⚠️ No documents fetched. Check your internet connection.")
        print("   PubMed API requires internet access.")
        return
    domain_counts = {}
    for doc in all_documents:
        domain = doc["tags"][0] if doc["tags"] else "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    print("\n  Breakdown by domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"    • {domain}: {count} documents")
    if os.path.exists(OUTPUT_PATH):
        backup_path = OUTPUT_PATH.replace(".json", "_backup.json")
        print(f"\n  💾 Backing up existing file to: {backup_path}")
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            old_data = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(old_data)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved to: {OUTPUT_PATH}")
    print(f"  🚀 Restart your app to load the new knowledge base.")
    print(f"{"=" * 60}")

if __name__ == "__main__":
    main()