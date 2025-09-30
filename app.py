# app.py (final - taxonomy cleaned)
import streamlit as st
import pickle
import faiss
import numpy as np
from Bio import SeqIO
import io
import re
import pandas as pd
import os

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH   = "ssu_faiss.index"        # adjust if needed
VECTORIZER_PATH    = "ssu_vectorizer.pkl"
IDS_PATH           = "ssu_ids.pkl"
TAX_PATH           = "SSU_taxonomy.txt"      # <-- using your taxonomy .txt
DEFAULT_KMER       = 6
DEFAULT_TOPK       = 5
DEFAULT_THRESHOLD  = 1.2   # distance threshold
# ----------------------------------------------------------------

def split_tokens(s): return s.split()

st.set_page_config(page_title="🌊 eDNA Eukaryotic Dashboard", layout="wide")
st.title("🌊 eDNA Eukaryotic Dashboard")
st.markdown("📊 Explore **species diversity** from Eukaryotic database and classify new sequences.")

# ---------------- Load artifacts ----------------
@st.cache_resource(ttl=3600)
def load_artifacts():
    # Load FAISS + Vectorizer + IDs
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    with open(IDS_PATH, "rb") as f: ids = pickle.load(f)

    # Load taxonomy from TXT
    tax_map = {}
    if os.path.exists(TAX_PATH):
        with open(TAX_PATH, "r") as f:
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


# ----------------- ensure same tokenizer exists for unpickling -----------------
def split_tokens(s):
    return s.split()

# ---------------- Load taxonomy (.txt -> dict) ----------------
def load_taxonomy_txt(path):
    """
    Load taxonomy from a .txt file with format:
    <SeqID> <Taxonomy description...>
    Extract only the first 2 words after ID (scientific name).
    """
    tax_map = {}
    if not os.path.exists(path):
        return tax_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                seq_id, desc = parts
                # take only first 2 words (Genus + species)
                sci_name = " ".join(desc.split()[:2])
                tax_map[seq_id] = sci_name
            else:
                tax_map[parts[0]] = "Unknown taxonomy"
    return tax_map

# ---------------- Load artifacts ----------------
@st.cache_resource(ttl=3600)
def load_artifacts():
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index not found at {FAISS_INDEX_PATH}")
        return None
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        st.error(f"Failed to read FAISS index: {e}")
        return None
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load vectorizer: {e}")
        return None
    try:
        with open(IDS_PATH, "rb") as f:
            ids = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load ids: {e}")
        return None

    tax_map = load_taxonomy_txt(TAX_PATH)

    return {"index": index, "vectorizer": vectorizer, "ids": ids, "tax_map": tax_map}

art = load_artifacts()
if art is None:
    st.stop()

index = art["index"]
vectorizer = art["vectorizer"]
ids = art["ids"]
tax_map = art["tax_map"]

# ---------------- Helpers ----------------
def clean_seq(s):
    return re.sub(r'[^ACGT]', '', s.upper())

def seq_to_kmers(seq, k=6):
    seq = seq.upper()
    if len(seq) < k:
        return ""
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    kmers = [km for km in kmers if all(ch in "ACGT" for ch in km)]
    return " ".join(kmers)

def interpret_distance(dist, thresholds=(1.0, 1.5, 2.0)):
    if dist is None:
        return "N/A"
    if dist <= thresholds[0]:
        return "Very close — likely same species"
    if dist <= thresholds[1]:
        return "Close — likely same genus"
    if dist <= thresholds[2]:
        return "Distant — possible novel lineage"
    return "Very distant — novel/unknown"

def classify_query(seq, kmer, top_k, threshold):
    """
    Return:
      - label
      - best_distance
      - neighbors
    """
    default_resp = {"label": "Unknown", "best_distance": None, "neighbors": []}

    kmers = seq_to_kmers(seq, k=kmer)
    if not kmers:
        default_resp.update({"error": "Sequence too short or invalid after cleaning."})
        return default_resp

    try:
        emb = vectorizer.transform([kmers]).toarray().astype("float32")
    except Exception as e:
        default_resp.update({"error": f"Vectorizer transform failed: {e}"})
        return default_resp

    try:
        D, I = index.search(emb, top_k)
    except Exception as e:
        default_resp.update({"error": f"FAISS search failed: {e}"})
        return default_resp

    neighbors = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0 or idx >= len(ids):
            continue
        ref_id = ids[idx]
        tax = tax_map.get(ref_id, "Unknown")
        neighbors.append({"rank": rank, "ref_id": ref_id, "distance": float(dist), "taxonomy": tax})

    if neighbors:
        best_dist = neighbors[0]["distance"]
        label = "Known (close match)" if (best_dist is not None and best_dist <= threshold) else "Novel candidate"
        return {"label": label, "best_distance": best_dist, "neighbors": neighbors}
    else:
        return {"label": "No matches found", "best_distance": None, "neighbors": []}

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("⚙️ Parameters")
    kmer = st.number_input("k-mer length (k)", value=DEFAULT_KMER, min_value=3, max_value=12, step=1)
    top_k = st.number_input("Top matches", value=DEFAULT_TOPK, min_value=1, max_value=20, step=1)
    threshold = st.slider("Novelty threshold (distance)", min_value=0.0, max_value=5.0, value=float(DEFAULT_THRESHOLD), step=0.01)
    st.caption("Distance ≤ threshold → Known; > threshold → Novel candidate")

# ---------------- Input: Sequence ----------------
st.subheader("Input DNA Sequence")
col1, col2 = st.columns([2,1])

with col1:
    seq_text = st.text_area(
        "Paste a DNA sequence (ACGT only)", 
        value=st.session_state.get("seq_input", ""),  # <-- pre-fill
        height=150
    )
    seq_file = st.file_uploader("Or upload a FASTA file", type=["fa", "fasta", "fas"])

with col2:
    if st.button("Use example sequence"):
        st.session_state["seq_input"] = "AGCCAGGCATGTCTAGTACAAACCCCAAAGGGGGAAAACCGCGAAAGGCTCATTAAATCAGTTATGTTTCCTTTGATTGGAACCTTTTTTTTGATAACGGTGGTAATTCTAGAGCAAATA"
        st.rerun()
# ---------------- Choose input priority ----------------
seq_input = None

if seq_file:  # ✅ uploaded FASTA
    try:
        text = seq_file.getvalue().decode('utf-8')
        fasta_io = io.StringIO(text)
        recs = list(SeqIO.parse(fasta_io, "fasta"))
        if len(recs) > 0:
            seq_input = str(recs[0].seq)
            st.success(f"Loaded FASTA record: {recs[0].id}")
    except Exception as e:
        st.error("Failed to read uploaded FASTA: " + str(e))
        st.stop()
elif "seq_input" in st.session_state:  # ✅ example button
    seq_input = st.session_state["seq_input"]
elif seq_text.strip():  # ✅ user-typed sequence
    seq_input = seq_text.strip()

# # ---------------- Input: Taxonomy ----------------
# st.subheader("Optional: Upload Taxonomy File")
# tax_file = st.file_uploader("Upload taxonomy .txt file", type=["txt"])

# if tax_file:
#     try:
#         text = tax_file.getvalue().decode('utf-8').splitlines()
#         tax_map = {}
#         for line in text:
#             parts = line.strip().split(maxsplit=1)
#             if len(parts) == 2:
#                 seq_id, desc = parts
#                 sci_name = " ".join(desc.split()[:2])  # only first 2 words
#                 tax_map[seq_id] = sci_name
#         st.success(f"Loaded taxonomy for {len(tax_map)} sequences.")
#     except Exception as e:
#         st.error("Failed to parse taxonomy file: " + str(e))
#         tax_map = {}
# else:
#     tax_map = art.get("tax_map", {})  # fallback to default

# ---------------- Run only when user clicks ----------------
run_btn = st.button("🔍 Classify Sequence")

if not seq_input:
    st.info("Paste/upload a sequence and click **Classify Sequence**.")
    st.stop()

if not run_btn:
    st.info("Click **Classify Sequence** to run.")
    st.stop()



# ---------------- Run classification ----------------
seq_clean = clean_seq(seq_input)
if len(seq_clean) < kmer:
    st.error(f"Sequence too short after cleaning ({len(seq_clean)} bases). Need ≥ {kmer}.")
    st.stop()

with st.spinner("Running classification..."):
    res = classify_query(seq_clean, kmer, top_k, threshold)

if res.get("error"):
    st.error(res["error"])
    st.stop()

# ---------------- Results ----------------
st.subheader("Result")
colA, colB = st.columns([2,1])
with colA:
    label = res.get("label", "Unknown")
    best_dist = res.get("best_distance", None)
    st.markdown(f"**Label:** `{label}`")
    if best_dist is None:
        st.markdown("**Best distance:** `N/A`")
        st.markdown("**Interpretation:** N/A")
    else:
        st.markdown(f"**Best distance:** `{best_dist:.4f}`")
        st.markdown(f"**Interpretation:** {interpret_distance(best_dist)}")
with colB:
    st.markdown("**Summary**")
    st.write(f"Seq length: {len(seq_clean)} bp")
    st.write(f"k-mer size: {kmer}")
    st.write(f"Top k: {top_k}")

neighbors = res.get("neighbors", [])
if neighbors:
    df = pd.DataFrame(neighbors)
    df = df.rename(columns={"ref_id":"Reference ID", "taxonomy":"Taxonomy", "distance":"Distance", "rank":"Rank"})
    st.dataframe(df)
else:
    st.warning("No neighbor matches were returned for this query.")

st.markdown("---")
st.caption("Prototype using TF-IDF k-mer embeddings + FAISS. For production → replace with DNABERT embeddings + benchmark against QIIME2/DADA2.")
