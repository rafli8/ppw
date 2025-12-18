import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import re
import nltk
import requests
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Download NLTK resources ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==========================================
# 1. FUNGSI HELPER
# ==========================================

def download_pdf_from_arxiv(paper_id):
    """Mendownload PDF dengan Debugging Error."""
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://arxiv.org/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return io.BytesIO(response.content)
        else:
            st.error(f"Gagal download dari ArXiv. Status Code: {response.status_code}")
            st.error("Kemungkinan ID salah atau ArXiv memblokir sementara.")
            return None
            
    except Exception as e:
        st.error(f"Terjadi error koneksi: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Mengekstrak teks mentah dari file object PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extract = page.extract_text()
            if extract:
                text += extract + " "
        return text
    except Exception as e:
        return None

def clean_and_tokenize(text):
    """Membersihkan teks, menghapus stopwords, dan tokenisasi."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english')) 
    custom_stops = {'et', 'al', 'fig', 'table', 'data', 'using', 'based', 'model', 'results', 'show', 'proposed'} 
    stop_words.update(custom_stops)
    
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return filtered_tokens

def build_cooccurrence_graph(tokens, window_size=3):
    """Membangun graph co-occurrence."""
    graph = nx.Graph()
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        for j in range(len(window)):
            for k in range(j + 1, len(window)):
                w1, w2 = sorted([window[j], window[k]])
                if w1 != w2:
                    if graph.has_edge(w1, w2):
                        graph[w1][w2]['weight'] += 1
                    else:
                        graph.add_edge(w1, w2, weight=1)
    return graph

# ==========================================
# 2. VISUALISASI (Update: Parameter threshold dihapus)
# ==========================================

def plot_graph(graph, pagerank_scores):
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(graph, k=0.15, iterations=50, seed=42)
    
    # Gambar Node
    node_sizes = [v * 20000 for v in pagerank_scores.values()]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    
    # Gambar Edge
    edges = graph.edges(data=True)
    weights = [data['weight'] for u, v, data in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [(w / max_weight) * 3 for w in weights]
    
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
    
    # Gambar Label
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight='bold', ax=ax)
    
    ax.axis('off')
    return fig

# ==========================================
# 3. INTERFACE UTAMA
# ==========================================

st.set_page_config(layout="wide", page_title="ArXiv Keyword Graph")

st.title("ArXiv Graph Analyzer")
st.markdown("Masukkan ID Paper arXiv untuk melihat visualisasi hubungan kata kuncinya.")

# Sidebar Settings
with st.sidebar:
    st.header("Input Paper")
    
    input_arxiv_id = st.text_input("Masukkan ArXiv ID:", value="1706.03762")
    st.caption("Contoh ID: 1706.03762 (Attention Is All You Need)")
    
    st.divider()
    
    st.header("Graph Settings")
    window_size = st.slider("Window Size", 2, 5, 3)
    top_n_words = st.slider("Jumlah Keyword", 5, 20, 10)
    
    # REVISI: Slider Threshold dihapus total di sini
    
    process_btn = st.button("Analisis Paper")

# Logika Proses
if input_arxiv_id:
    
    with st.spinner(f'Mendownload Paper ID: {input_arxiv_id}...'):
        pdf_file = download_pdf_from_arxiv(input_arxiv_id)
    
    if pdf_file:
        raw_text = extract_text_from_pdf(pdf_file)
        
        if raw_text:
            tokens = clean_and_tokenize(raw_text)
            st.success(f"Berhasil mengambil paper! Ditemukan {len(tokens)} kata relevan.")
            
            if len(tokens) > 0:
                with st.spinner('Menghitung PageRank...'):
                    graph = build_cooccurrence_graph(tokens, window_size=window_size)
                    pagerank_scores = nx.pagerank(graph, weight='weight')
                    sorted_keywords = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Top Keywords")
                    df_keywords = pd.DataFrame(sorted_keywords[:top_n_words], columns=["Kata", "Score"])
                    st.dataframe(df_keywords, use_container_width=True)
                    
                with col2:
                    st.subheader("Visualisasi Graph")
                    # REVISI: Pemanggilan fungsi tanpa parameter threshold
                    fig = plot_graph(graph, pagerank_scores)
                    st.pyplot(fig)
            else:
                st.warning("Teks paper terlalu pendek atau tidak terbaca.")
        else:
            st.error("Gagal mengekstrak teks dari PDF. Mungkin PDF berisi gambar scan.")
    else:
        st.error(f"Gagal mendownload paper dengan ID {input_arxiv_id}. Cek kembali ID-nya.")
