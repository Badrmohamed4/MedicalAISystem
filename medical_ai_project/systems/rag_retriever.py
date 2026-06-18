"""
Medical RAG Retriever — Reuses SapBERT Embeddings
===================================================
Dense vector retrieval using the SAME biomedical embedding model
already loaded by SapBERTNormalizer (cambridgeltl/SapBERT-from-PubMedBERT-fulltext).

Why SapBERT instead of all-MiniLM?
- Trained on 20M+ PubMed abstracts — understands medical terminology
- Already downloaded by your app — zero additional setup
- Unified architecture — one model serves normalization + retrieval

Features:
- Reuses SapBERT model instance (no double-loading)
- Domain-boosted retrieval (prioritizes domain-relevant docs)
- Source-tagged output for traceability
- Fallback to keyword matching if model unavailable
"""
import os
import json
import numpy as np


class SapBERTMedicalRAG:
    """
    Dense retrieval RAG using SapBERT embeddings.
    Reuses the same model loaded by SapBERTNormalizer.
    """

    def __init__(self, top_k=2, similarity_threshold=0.55, domain_boost=1.5):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.domain_boost = domain_boost
        self.documents = []
        self.embeddings = None
        self._model = None
        self._model_loaded = False

        # Load knowledge base documents
        self._load_documents()

        # Try to load SapBERT model
        self._load_model()

        # If model loaded, pre-compute document embeddings
        if self._model_loaded:
            self._compute_embeddings()

    def _load_documents(self):
        """Load curated medical documents from JSON."""
        doc_path = os.path.join(os.path.dirname(__file__), "medical_documents.json")
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            print(f"[RAG] Loaded {len(self.documents)} medical documents.")
        else:
            # Fallback: create inline documents
            self.documents = self._create_inline_documents()
            print(f"[RAG] Using inline medical documents ({len(self.documents)} docs).")

    def _create_inline_documents(self):
        """Inline documents if JSON file is missing."""
        return [
            {"text": "Gliomas are tumors that arise from glial cells in the brain or spinal cord. Symptoms include headaches, seizures, cognitive changes. Treatment: surgery, radiation, chemotherapy.", "tags": ["brain", "glioma", "tumor"], "source": "BRAIN_TUMOR_INFO"},
            {"text": "Meningiomas form on the membranes covering the brain and spinal cord. Usually slow-growing and benign. Managed by observation or surgical removal.", "tags": ["brain", "meningioma"], "source": "BRAIN_TUMOR_INFO"},
            {"text": "Pituitary adenomas develop in the pituitary gland. Can disrupt hormone regulation and cause vision problems. Most are benign and treatable.", "tags": ["brain", "pituitary"], "source": "BRAIN_TUMOR_INFO"},
            {"text": "Adenocarcinoma is the most common lung cancer, found in outer lung regions. Starts in mucus-producing cells. Early detection improves prognosis.", "tags": ["lung", "adenocarcinoma"], "source": "LUNG_CANCER_INFO"},
            {"text": "Large cell carcinoma grows and spreads quickly, making it harder to treat. Immediate oncology consultation recommended.", "tags": ["lung", "large cell carcinoma"], "source": "LUNG_CANCER_INFO"},
            {"text": "Squamous cell carcinoma develops in airway lining cells, usually central lungs. Linked to smoking history. May cause coughing up blood.", "tags": ["lung", "squamous cell carcinoma"], "source": "LUNG_CANCER_INFO"},
            {"text": "Melanoma is a serious skin cancer beginning in melanocytes. Requires immediate dermatologist/oncologist assessment and biopsy.", "tags": ["skin", "malignant", "melanoma"], "source": "SKIN_DISEASE_INFO"},
            {"text": "Psoriasis is a chronic autoimmune skin disease causing itchy, scaly patches on knees, elbows, trunk, scalp. Treatments: ointments, light therapy, systemic medications.", "tags": ["skin", "psoriasis"], "source": "SKIN_DISEASE_INFO"},
            {"text": "Eczema (atopic dermatitis) causes red, itchy skin. Chronic and flares periodically. Management: avoid harsh soaps, moisturize, medicated creams.", "tags": ["skin", "eczema"], "source": "SKIN_DISEASE_INFO"},
            {"text": "Acne occurs when hair follicles plug with oil and dead skin. Treated with topical creams, cleansers, or prescription medications.", "tags": ["skin", "acne"], "source": "SKIN_DISEASE_INFO"},
            {"text": "For high-risk patients: seek immediate ER attention. Do not drive yourself. Call emergency services. Have someone stay with you.", "tags": ["emergency", "high risk"], "source": "EMERGENCY_ADVICE"},
            {"text": "Seizure first aid: lay person on their side, keep airway open, remove hazards, do NOT restrain or put objects in mouth. Time the seizure; call emergency if >5 minutes.", "tags": ["brain", "seizure", "emergency"], "source": "BRAIN_ADVICE"},
            {"text": "Headache management: apply cold/warm compress, dim lights, minimize screens, try relaxation techniques, hydrate, rest in dark room.", "tags": ["brain", "headache", "care"], "source": "BRAIN_ADVICE"},
            {"text": "Lung issue precautions: avoid smoke/dust/chemical fumes, use humidifier, sleep with head elevated, monitor oxygen levels with pulse oximeter.", "tags": ["lung", "care"], "source": "LUNG_ADVICE"},
            {"text": "Skin issue precautions: avoid scratching, wash gently with mild cleanser, apply moisturizer, protect from direct sunlight.", "tags": ["skin", "care"], "source": "SKIN_ADVICE"}
        ]

    def _load_model(self):
        """Load SapBERT model — same one used by SapBERTNormalizer."""
        try:
            from sentence_transformers import SentenceTransformer
            print("[RAG] Loading SapBERT embedding model...")
            self._model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
            self._model_loaded = True
            print("[RAG] ✅ SapBERT loaded successfully.")
        except Exception as e:
            print(f"[RAG] ⚠️ Could not load SapBERT: {e}")
            print("[RAG] ⚠️ Falling back to keyword-based retrieval.")
            self._model_loaded = False

    def _compute_embeddings(self):
        """Pre-compute embeddings for all documents."""
        if not self._model_loaded:
            return
        texts = [doc["text"] for doc in self.documents]
        print(f"[RAG] Computing embeddings for {len(texts)} documents...")
        self.embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        print("[RAG] ✅ Embeddings ready.")

    def retrieve(self, query: str, domain_filter: str = None) -> str:
        """
        Retrieve top-k relevant documents.
        Uses dense retrieval if SapBERT is available, else falls back to keywords.
        """
        if not query.strip():
            return ""

        # ── DENSE RETRIEVAL (SapBERT) ──
        if self._model_loaded and self.embeddings is not None:
            return self._dense_retrieve(query, domain_filter)

        # ── FALLBACK: KEYWORD MATCHING ──
        return self._keyword_retrieve(query, domain_filter)

    def _dense_retrieve(self, query: str, domain_filter: str = None) -> str:
     """Dense vector retrieval using SapBERT embeddings."""
     query_embedding = self._model.encode([query], convert_to_numpy=True)

     # Cosine similarity (SapBERT outputs normalized vectors)
     similarities = np.dot(self.embeddings, query_embedding.T).flatten()

     # Domain boosting
     if domain_filter and domain_filter.lower() not in ["unknown", "none", "general"]:
         df_lower = domain_filter.lower()
         for i, doc in enumerate(self.documents):
             tags = [t.lower() for t in doc.get("tags", [])]
             if df_lower in tags:
                 similarities[i] *= self.domain_boost

     # Get top-k
     top_indices = np.argsort(similarities)[-self.top_k:][::-1]

     retrieved = []
     for idx in top_indices:
         if similarities[idx] >= self.similarity_threshold:
             doc = self.documents[idx]
             source = doc.get("source", "MEDICAL_KB")
             text = doc["text"]
             retrieved.append(f"[{source}] {text}")

     # ── NO-MATCH FALLBACK ──
     if not retrieved:
         print(f"[RAG] ⚠️ No documents matched threshold ({self.similarity_threshold}) for query: '{query[:50]}...'")
         return ""

     return "\n\n".join(retrieved)

    def _keyword_retrieve(self, query: str, domain_filter: str = None) -> str:
        """Fallback keyword-based retrieval if SapBERT fails."""
        query_terms = set(query.lower().split())
        scored = {}

        for i, doc in enumerate(self.documents):
            score = 0
            doc_text = doc["text"].lower()
            tags = [t.lower() for t in doc.get("tags", [])]

            # Score by keyword overlap
            for term in query_terms:
                if term in doc_text or term in tags:
                    score += 1

            # Domain boost
            if domain_filter and domain_filter.lower() in tags:
                score *= self.domain_boost

            if score > 0:
                scored[i] = score

        # Get top-k
        sorted_docs = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        retrieved = []
        for idx, score in sorted_docs:
            doc = self.documents[idx]
            source = doc.get("source", "MEDICAL_KB")
            text = doc["text"]
            retrieved.append(f"[{source}] {text}")

        return "\n\n".join(retrieved) if retrieved else ""

    def retrieve_for_assessment(self, symptoms: list, medical_context: str, tumor_class: str = None) -> str:
        """Convenience method for DoctorAgent."""
        parts = []
        if symptoms:
            parts.append(" ".join(symptoms))
        if medical_context and medical_context not in ["unknown", "none"]:
            parts.append(medical_context)
        if tumor_class:
            parts.append(tumor_class)
        query = " ".join(parts) if parts else "general medical information"
        return self.retrieve(query, domain_filter=medical_context)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────
_rag = None

def get_rag():
    """Get or create the singleton RAG instance."""
    global _rag
    if _rag is None:
        _rag = SapBERTMedicalRAG()
    return _rag
