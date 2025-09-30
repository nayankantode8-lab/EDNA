# summary_dashboard_txt.py - eDNA SSU Summary + Classifier (using SSU_taxonomy.txt)

import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import os
import re

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH   = "ssu_faiss.index"
VECTORIZER_PATH    = "ssu_vectorizer.pkl"
IDS_PATH           = "ssu_ids.pkl"
TAXONOMY_TXT_PATH  = "SSU_taxonomy.txt"   # <-- your TXT file
DEFAULT_KMER       = 6
DEFAULT_TOPK       = 5
DEFAULT_THRESHOLD  = 1.2
# ----------------------------------------------------------------

def split_tokens(s): return s.split()

st.set_page_config(page_title="🌊 eDNA SSU Dashboard", layout="wide")
st.title("🌊 eDNA SSU Dashboard")
st.markdown("📊 Explore **species diversity** from SSU database and classify new sequences.")

# ---------------- Load artifacts ----------------
@st.cache_resource(ttl=3600)
def load_artifacts():
    # Load FAISS + Vectorizer + IDs
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    with open(IDS_PATH, "rb") as f: ids = pickle.load(f)

    # Load taxonomy from TXT
    tax_map = {}
    if os.path.exists(TAXONOMY_TXT_PATH):
        with open(TAXONOMY_TXT_PATH, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    tax_map[parts[0]] = parts[1] + " " + parts[2]  # only species name
    return {"index": index, "vectorizer": vectorizer, "ids": ids, "tax_map": tax_map}

art = load_artifacts()
index, vectorizer, ids, tax_map = art["index"], art["vectorizer"], art["ids"], art["tax_map"]

# ---------------- Summary FIRST ----------------
st.subheader("📊 SSU Reference Species Summary")

if tax_map:
    df_tax = pd.DataFrame(list(tax_map.items()), columns=["ID", "Species"])
    abundance = df_tax["Species"].value_counts().reset_index()
    abundance.columns = ["Species", "Count"]

    st.write("### Abundance Matrix")
    st.dataframe(abundance)

    # Diversity indices
    total = abundance["Count"].sum()
    p = abundance["Count"] / total
    shannon = -(p * np.log(p)).sum()
    simpson = 1 - (p**2).sum()
    chao1 = len(abundance) + (abundance["Count"].eq(1).sum()**2) / (2*abundance["Count"].eq(2).sum() + 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observed Species", len(abundance))
    col2.metric("Shannon Index (H′)", f"{shannon:.3f}")
    col3.metric("Simpson Index (1-D)", f"{simpson:.3f}")
    col4.metric("Chao1 Estimator", f"{chao1:.1f}")

    # Bar chart of top 15 species
    st.bar_chart(abundance.set_index("Species").head(15))
else:
    st.warning("⚠️ Taxonomy TXT file not found or empty.")

st.markdown("---")

# ---------------- Classifier ----------------
st.subheader("🔍 Classify a New DNA Sequence")

def clean_seq(s): return re.sub(r'[^ACGT]', '', s.upper())

def seq_to_kmers(seq, k=6):
    if len(seq) < k: return ""
    return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])

def classify_query(seq, kmer, top_k, threshold):
    kmers = seq_to_kmers(seq, k=kmer)
    if not kmers: return {"label": "Invalid", "neighbors": []}
    emb = vectorizer.transform([kmers]).toarray().astype("float32")
    D, I = index.search(emb, top_k)
    neighbors = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0 or idx >= len(ids): continue
        ref_id = ids[idx]
        tax = tax_map.get(ref_id, "Unknown")
        neighbors.append({"Rank": rank, "Reference ID": ref_id, "Distance": float(dist), "Species": tax})
    label = "Known" if neighbors and neighbors[0]["Distance"] <= threshold else "Novel"
    return {"label": label, "neighbors": neighbors}

# Sidebar parameters
with st.sidebar:
    st.header("⚙️ Parameters")
    kmer = st.number_input("k-mer length (k)", value=DEFAULT_KMER, min_value=3, max_value=12, step=1)
    top_k = st.number_input("Top matches", value=DEFAULT_TOPK, min_value=1, max_value=20, step=1)
    threshold = st.slider("Novelty threshold", 0.0, 5.0, float(DEFAULT_THRESHOLD), 0.01)

# Input box
seq_input = st.text_area("Paste DNA sequence (ACGT only)", height=120)

if st.button("Classify Sequence"):
    if not seq_input.strip():
        st.warning("Enter a DNA sequence.")
    else:
        seq_clean = clean_seq(seq_input)
        res = classify_query(seq_clean, kmer, top_k, threshold)
        st.write(f"**Label:** {res['label']}")
        if res["neighbors"]:
            df = pd.DataFrame(res["neighbors"])
            st.dataframe(df)
    