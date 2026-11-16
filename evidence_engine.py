
import re
import numpy as np

from collections import OrderedDict
from google import genai
from google.genai import types


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDING_MODEL = "gemini-embedding-2"

MAX_CACHE_ENTRIES = 8

# Stores reusable source embeddings.
# Cache remains available while the Python process runs.
_SOURCE_EMBEDDING_CACHE = OrderedDict()


# ==========================================
# CLIENT
# ==========================================

def create_client(api_key):

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    return genai.Client(api_key=api_key)


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(client, texts):

    cleaned_texts = [
        clean_text(text)
        for text in texts
        if clean_text(text)
    ]

    if not cleaned_texts:
        return []

    contents = [
        types.Content(
            parts=[
                types.Part.from_text(text=text)
            ]
        )
        for text in cleaned_texts
    ]

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents
    )

    if not response.embeddings:
        raise ValueError(
            "Gemini returned no embeddings."
        )

    vectors = [
        np.asarray(
            item.values,
            dtype=np.float32
        )
        for item in response.embeddings
    ]

    if len(vectors) != len(cleaned_texts):

        raise ValueError(
            f"Embedding count mismatch: "
            f"{len(cleaned_texts)} texts submitted, "
            f"{len(vectors)} embeddings returned."
        )

    return vectors


# ==========================================
# SOURCE EMBEDDING CACHE
# ==========================================

def get_source_embeddings(client, source_texts):

    source_texts = [
        clean_text(text)
        for text in source_texts
        if clean_text(text)
    ]

    if not source_texts:
        return []

    cache_key = (
        EMBEDDING_MODEL,
        tuple(source_texts)
    )

    # Reuse existing embeddings when available.
    if cache_key in _SOURCE_EMBEDDING_CACHE:

        _SOURCE_EMBEDDING_CACHE.move_to_end(
            cache_key
        )

        return _SOURCE_EMBEDDING_CACHE[
            cache_key
        ]

    # Generate source embeddings only once.
    vectors = generate_embeddings(
        client,
        source_texts
    )

    _SOURCE_EMBEDDING_CACHE[cache_key] = vectors

    # Limit memory usage.
    while len(_SOURCE_EMBEDDING_CACHE) > MAX_CACHE_ENTRIES:

        _SOURCE_EMBEDDING_CACHE.popitem(
            last=False
        )

    return vectors


# ==========================================
# CLEAR CACHE
# ==========================================

def clear_embedding_cache():

    _SOURCE_EMBEDDING_CACHE.clear()


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)

    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    score = np.dot(
        vector_a,
        vector_b
    ) / (norm_a * norm_b)

    return float(score)


# ==========================================
# RANK RETRIEVED EVIDENCE
# ==========================================

def rank_evidence(
    query_vector,
    source_texts,
    source_vectors,
    top_k=3
):

    ranked_results = []

    for fact, vector in zip(
        source_texts,
        source_vectors
    ):

        score = cosine_similarity(
            query_vector,
            vector
        )

        ranked_results.append({

            "text": fact,

            "similarity": round(
                score * 100,
                2
            )

        })

    ranked_results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return ranked_results[:top_k]


# ==========================================
# SEMANTIC RETRIEVAL
# ==========================================

def retrieve_evidence(
    client,
    query,
    source_facts,
    top_k=3
):

    source_facts = [
        clean_text(fact)
        for fact in source_facts
        if clean_text(fact)
    ]

    query = clean_text(query)

    if not source_facts or not query:
        return []

    # Reuse cached source embeddings.
    source_vectors = get_source_embeddings(
        client,
        source_facts
    )

    # Generate only the new query embedding.
    query_vectors = generate_embeddings(
        client,
        [query]
    )

    return rank_evidence(
        query_vectors[0],
        source_facts,
        source_vectors,
        top_k
    )


# ==========================================
# CONTRADICTION CANDIDATE DETECTION
# ==========================================

def detect_contradiction_candidates(
    client,
    claim,
    restrictions,
    top_k=3
):

    if not restrictions:
        return []

    return retrieve_evidence(
        client=client,
        query=claim,
        source_facts=restrictions,
        top_k=top_k
    )


# ==========================================
# COMPLETE EVIDENCE ANALYSIS
# ==========================================

def analyze_evidence(
    client,
    claim,
    approved_facts,
    restrictions
):

    claim = clean_text(claim)

    approved_facts = [
        clean_text(fact)
        for fact in approved_facts
        if clean_text(fact)
    ]

    restrictions = [
        clean_text(item)
        for item in restrictions
        if clean_text(item)
    ]

    if not claim:

        return {
            "claim": claim,
            "approved_evidence": [],
            "restriction_matches": []
        }

    # Combine both source groups.
    # Their embeddings can be reused across sentences.
    all_sources = approved_facts + restrictions

    if not all_sources:

        return {
            "claim": claim,
            "approved_evidence": [],
            "restriction_matches": []
        }

    # Retrieve cached source embeddings.
    all_source_vectors = get_source_embeddings(
        client,
        all_sources
    )

    # Generate one query embedding for this claim.
    query_vectors = generate_embeddings(
        client,
        [claim]
    )

    query_vector = query_vectors[0]

    approved_count = len(approved_facts)

    approved_vectors = all_source_vectors[
        :approved_count
    ]

    restriction_vectors = all_source_vectors[
        approved_count:
    ]

    # Rank approved facts.
    approved_matches = rank_evidence(
        query_vector=query_vector,
        source_texts=approved_facts,
        source_vectors=approved_vectors,
        top_k=3
    )

    # Rank potential restriction matches.
    restriction_matches = rank_evidence(
        query_vector=query_vector,
        source_texts=restrictions,
        source_vectors=restriction_vectors,
        top_k=3
    )

    return {

        "claim": claim,

        "approved_evidence": approved_matches,

        "restriction_matches": restriction_matches

    }
