"""
CampusSeek — Sistem Navigasi Ruangan Kampus Berbasis NLP
=========================================================
Judul Penelitian:
  Aplikasi Web Sistem Navigasi Ruangan Kampus Berbasis Natural Language
  Processing Menggunakan IndoBERT dengan Klasifikasi Intent, Named Entity
  Recognition, dan Panduan Arah Berbasis Jarak Algoritma A-Star

Jalankan: streamlit run app.py
Requirement: pip install streamlit transformers torch
"""

import re, csv, math, heapq
import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

import torch
import pickle

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CampusSeek — Navigasi Kampus",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;}
.banner{background:linear-gradient(135deg,#1B3A6B 0%,#2E5FA3 100%);
  padding:1.4rem 2rem;border-radius:14px;margin-bottom:1rem;color:white;}
.banner h1{font-size:1.9rem;margin:0;font-weight:700;}
.banner p{margin:.25rem 0 0;opacity:.85;font-size:.9rem;}
.stTextInput>div>div>input{border-radius:30px!important;
  border:2px solid #2E5FA3!important;padding:.6rem 1.2rem!important;font-size:1rem!important;}
.room-card{background:#f0f6ff;border:1px solid #cce0ff;
  border-left:5px solid #2E5FA3;border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem;}
.room-card h3{margin:0 0 .3rem;color:#1B3A6B;font-size:1.05rem;}
.room-card p{margin:0;color:#444;font-size:.86rem;}
.step-box{display:flex;align-items:flex-start;gap:.8rem;background:white;
  border:1px solid #e0e8f0;border-radius:10px;padding:.7rem 1rem;
  margin-bottom:.45rem;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.step-num{background:#2E5FA3;color:white;border-radius:50%;width:28px;height:28px;
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.8rem;flex-shrink:0;}
.step-last .step-num{background:#1A7A4A;}
.step-text{color:#222;font-size:.9rem;line-height:1.4;padding-top:3px;}
.nlp-badge{display:inline-block;padding:2px 10px;border-radius:20px;
  font-size:.76rem;font-weight:600;margin:2px 4px 4px 0;}
.b-intent{background:#e8f0fe;color:#1558d6;}
.b-ner{background:#e6f4ea;color:#1A7A4A;}
.b-lantai{background:#fde8e8;color:#B00020;}
.b-start{background:#e0f7fa;color:#006064;}
.b-conf{background:#f3e8ff;color:#6200ea;}
.metric-row{display:flex;gap:1rem;margin:.8rem 0;flex-wrap:wrap;}
.mc{background:white;border:1px solid #dde8f8;border-radius:10px;
  padding:.7rem 1.2rem;text-align:center;min-width:110px;flex:1;
  box-shadow:0 1px 4px rgba(0,0,0,.06);}
.mc .val{font-size:1.3rem;font-weight:700;color:#2E5FA3;}
.mc .lbl{font-size:.75rem;color:#666;margin-top:2px;}
.pos-ok{background:#e8f5e9;border:1px solid #a5d6a7;border-left:5px solid #2e7d32;
  border-radius:10px;padding:.7rem 1rem;margin-top:.4rem;font-size:.86rem;color:#1b5e20;}
.pos-warn{background:#fff3e0;border:1px solid #ffcc80;border-left:5px solid #ef6c00;
  border-radius:10px;padding:.7rem 1rem;margin-top:.4rem;font-size:.86rem;color:#bf360c;}
.note-box{background:#fffde7;border:1px solid #ffe082;border-radius:8px;
  padding:.45rem .85rem;font-size:.82rem;color:#5f4200;margin-top:6px;}
.nlp-panel{background:#f8f9ff;border:1px solid #dde4f5;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;}
.nlp-panel h4{margin:0 0 .5rem;color:#1B3A6B;font-size:.95rem;}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_nodes(path="nodes.csv"):
    nodes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["lantai"] = int(row["lantai"])
            row["x"]      = float(row["x"])
            row["y"]      = float(row["y"])
            row["kata_kunci"] = [k.strip().lower() for k in row["kata_kunci"].split(",")]
            nodes[row["node_id"]] = row
    return nodes

@st.cache_data
def load_edges(path="edges.csv"):
    graph = {}   # { node_id: [(neighbor_id, jarak), ...] }
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b, d = row["from_node"], row["to_node"], float(row["jarak_meter"])
            graph.setdefault(a, []).append((b, d))
            graph.setdefault(b, []).append((a, d))   # bidirectional
    return graph

@st.cache_data
def load_routes(path="routes.csv"):
    routes = {}  # { tujuan_node: row }
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["lantai_tujuan"]   = int(row["lantai_tujuan"])
            row["total_jarak_estimasi"] = int(row["total_jarak_estimasi"])
            row["langkah_list"]    = [s.strip() for s in row["langkah_navigasi"].split("|") if s.strip()]
            routes[row["tujuan_node"]] = row
    return routes

@st.cache_data
def load_intents(path="intent_dataset.csv"):
    intents = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            intents.append(row)
    return intents

try:
    nodes   = load_nodes()
    edges   = load_edges()
    routes  = load_routes()
    intents = load_intents()
except FileNotFoundError as e:
    st.error(f"❌ File CSV tidak ditemukan: {e}\n\nPastikan folder `data/` berisi:\n- nodes.csv\n- edges.csv\n- routes.csv\n- intent_dataset.csv")
    st.stop()

#indoBert
@st.cache_resource
def load_indobert_model():

    model_path = "triyogaprasetya/campussekk-intent"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    label_path = hf_hub_download(
        repo_id= "triyogaprasetya/campussekk-intent",
        filename="label_encoder.pkl"
    )

    with open(label_path, "rb") as f:
        encoder = pickle.load(f)

    return tokenizer, model

tokenizer_intent, model_intent, label_encoder = load_indobert_model()


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITMA A-STAR
# ══════════════════════════════════════════════════════════════════════════════
def heuristic(a: str, b: str, nodes: dict) -> float:
    """Euclidean distance antara dua node sebagai heuristik A-Star."""
    na, nb = nodes.get(a), nodes.get(b)
    if not na or not nb:
        return 0
    return math.sqrt((na["x"] - nb["x"])**2 + (na["y"] - nb["y"])**2)

def astar(start: str, goal: str, graph: dict, nodes: dict):
    """
    A-Star pathfinding.
    Return: { found, path, total_jarak, error }
    """
    if start == goal:
        return {"found": True, "path": [start], "total_jarak": 0}
    if start not in nodes or goal not in nodes:
        return {"found": False, "path": [], "total_jarak": 0,
                "error": f"Node '{start}' atau '{goal}' tidak ada di peta."}

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from  = {}
    g_score    = {start: 0}
    f_score    = {start: heuristic(start, goal, nodes)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Rekonstruksi jalur
            path, total = [], g_score[goal]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return {"found": True, "path": path, "total_jarak": round(total, 1)}

        for neighbor, weight in graph.get(current, []):
            tentative = g_score.get(current, float("inf")) + weight
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative
                f_score[neighbor]   = tentative + heuristic(neighbor, goal, nodes)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return {"found": False, "path": [], "total_jarak": 0,
            "error": "Jalur tidak ditemukan. Periksa koneksi peta."}


# ══════════════════════════════════════════════════════════════════════════════
# NLP PIPELINE  (Rule-based sebagai baseline, siap di-upgrade IndoBERT)
# ══════════════════════════════════════════════════════════════════════════════

# ── NER: Ekstraksi Entitas ────────────────────────────────────────────────────
def ner_extract(teks: str) -> dict:
    """
    Named Entity Recognition (NER) rule-based.
    Ekstrak: lantai, gedung, tipe_ruang, nama_prodi, nama_khusus
    """
    tl = teks.lower()
    entitas = {"lantai": None, "gedung": None, "tipe_ruang": None,
               "nama_prodi": None, "nama_khusus": None}

    # Lantai
    m = re.search(r"(?:lantai|lt\.?)\s*(\d)", tl)
    if m:
        entitas["lantai"] = int(m.group(1))

    # Gedung
    m = re.search(r"gedung\s*([a-zA-Z])", tl)
    if m:
        entitas["gedung"] = "Gedung " + m.group(1).upper()

    # Tipe ruang
    tipe_map = {
        "laboratorium": ["lab","laboratorium","praktikum"],
        "toilet":       ["toilet","wc","kamar mandi","sanitasi"],
        "perpustakaan": ["perpustakaan","perpus","library","buku"],
        "seminar":      ["seminar","sidang","presentasi"],
        "aula":         ["aula","serbaguna","acara"],
        "kantor":       ["kantor","kaprodi","dekan","tata usaha","administrasi"],
        "rapat":        ["rapat","meeting","senat"],
    }
    for tipe, kws in tipe_map.items():
        if any(kw in tl for kw in kws):
            entitas["tipe_ruang"] = tipe
            break

    # Nama prodi / bidang ilmu
    prodi_map = {
        "ilmu komputer": ["ilmu komputer","komputer","ilkom","informatika"],
        "kimia":         ["kimia"],
        "statistika":    ["statistika","statistik"],
        "biologi":       ["biologi"],
        "bioteknologi":  ["bioteknologi"],
        "matematika":    ["matematika","math","maths"],
        "fisika":        ["fisika"],
        "basis data":    ["basis data","database","sql","bd"],
        "genetika":      ["genetika","gen"],
        "zoologi":       ["zoologi","hewan","fauna"],
        "biokimia":      ["biokimia","biokimia dan bioteknologi"],
        "kimia organik": ["kimia organik","organik"],
        "kimia anorganik":["kimia anorganik","anorganik"],
        "kimia fisika":  ["kimia fisika","kimfis"],
    }
    for nama, kws in prodi_map.items():
        if any(kw in tl for kw in kws):
            entitas["nama_prodi"] = nama
            break

    # Nama khusus (dekan, UTI, dll)
    khusus_map = {
        "dekan": ["dekan","pimpinan fmipa"],
        "UTI":   ["uti","jaminan mutu","ujmsi"],
        "perlengkapan": ["perlengkapan","logistik"],
        "tata usaha":   ["tata usaha","tu","administrasi"],
        "senat":        ["senat","rapat senat"],
    }
    for nama, kws in khusus_map.items():
        if any(kw in tl for kw in kws):
            entitas["nama_khusus"] = nama
            break

    return entitas


# ── Intent Classification ─────────────────────────────────────────────────────
def classify_intent(teks, intents=None) -> dict:
    inputs = tokenizer_intent(
        teks,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model_intent(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    confidence, pred_idx = torch.max(
        probs,
        dim=1
    )

    label = label_encoder.inverse_transform(
        [pred_idx.item()]
    )[0]

    return {
        "intent_label": label,
        "node_target": None,
        "confidence": round(
            confidence.item(),
            2
        )
    }


# ── Pencarian Node Tujuan ─────────────────────────────────────────────────────
def cari_node_tujuan(query: str, nodes: dict, intents: list, entitas: dict) -> list:
    """
    Gabungkan hasil intent + NER + kata_kunci untuk mencari node tujuan.
    Return: list of (score, node_id, node_dict) diurutkan terbaik.
    """
    tl         = query.lower()
    kata_query = set(re.findall(r"\w+", tl))
    intent_res = classify_intent(query, intents)

    kandidat = {}

    for nid, node in nodes.items():
        if nid == "LOBBY":
            continue
        skor = 0.0
        nama_lower = node["nama_ruangan"].lower()

        # Bobot 1: Cocokkan kata_kunci node
        for kw in node["kata_kunci"]:
            if kw in tl:
                skor += 0.6
            elif any(k in kw for k in kata_query if len(k) > 3):
                skor += 0.25

        # Bobot 2: Token overlap ke nama ruangan
        kata_nama = set(re.findall(r"\w+", nama_lower))
        overlap_nama = len(kata_nama & kata_query) / max(len(kata_nama), 1)
        skor += overlap_nama * 0.5

        # Bobot 3: NER lantai
        if entitas["lantai"] is not None:
            skor += 0.4 if node["lantai"] == entitas["lantai"] else -0.3

        # Bobot 4: NER gedung
        if entitas["gedung"]:
            skor += 0.2 if entitas["gedung"].lower() in node["gedung"].lower() else -0.15

        # Bobot 5: NER tipe_ruang
        if entitas["tipe_ruang"] and entitas["tipe_ruang"] in node["tipe"].lower():
            skor += 0.4

        # Bobot 6: NER nama_prodi dalam nama_ruangan
        if entitas["nama_prodi"] and entitas["nama_prodi"] in nama_lower:
            skor += 0.7

        # Bobot 7: intent node_target cocok
        if intent_res["node_target"] == nid:
            skor += 0.8 * intent_res["confidence"]

        if skor > 0.1:
            kandidat[nid] = (skor, nid, node)

    hasil = sorted(kandidat.values(), key=lambda x: -x[0])
    return hasil, intent_res


# ── Resolve Posisi Awal ───────────────────────────────────────────────────────
def resolve_posisi_awal(teks: str, nodes: dict) -> dict:
    """
    NLP untuk input posisi awal pengguna.
    Return: { found, node_id, node, confidence, saran }
    """
    if not teks.strip():
        lobby = nodes.get("LOBBY", {})
        return {"found": False, "node_id": "LOBBY", "node": lobby, "confidence": 0.0, "saran": []}

    tl = teks.lower()
    entitas = ner_extract(teks)

    # Keyword tipe posisi awal
    tipe_kw = {
        "lobby":       ["lobby","pintu masuk","entrance","lobi","masuk","depan"],
        "tangga":      ["tangga","stairs","naik tangga"],
        "koridor":     ["koridor","lorong","gang","hallway"],
        "toilet":      ["toilet","wc","kamar mandi"],
        "perpustakaan":["perpustakaan","perpus"],
        "laboratorium":["lab","laboratorium"],
        "kantor":      ["kantor","kaprodi","dekan"],
    }
    tipe_target = None
    for tipe, kws in tipe_kw.items():
        if any(kw in tl for kw in kws):
            tipe_target = tipe
            break

    kandidat = []
    for nid, node in nodes.items():
        skor = 0.0
        nama_lower = node["nama_ruangan"].lower()
        kata_nama  = set(re.findall(r"\w+", nama_lower))
        kata_input = set(re.findall(r"\w+", tl))

        # Tipe cocok
        if tipe_target:
            if tipe_target in node["tipe"].lower():
                skor += 0.5
            elif any(kw in nama_lower for kw in tipe_kw.get(tipe_target, [])):
                skor += 0.35

        # Lantai
        if entitas["lantai"] is not None:
            skor += 0.35 if node["lantai"] == entitas["lantai"] else -0.2
        elif node["lantai"] == 1:
            skor += 0.1

        # Gedung
        if entitas["gedung"]:
            skor += 0.2 if entitas["gedung"].lower() in node["gedung"].lower() else -0.1

        # Token overlap nama
        ov = len(kata_nama & kata_input) / max(len(kata_nama), 1)
        skor += ov * 0.5

        # Kata_kunci cocok
        for kw in node["kata_kunci"]:
            if kw in tl:
                skor += 0.3

        # Prodi NER
        if entitas["nama_prodi"] and entitas["nama_prodi"] in nama_lower:
            skor += 0.6

        if skor > 0:
            kandidat.append((skor, nid, node))

    if not kandidat:
        lobby = nodes.get("LOBBY", {})
        return {"found": False, "node_id": "LOBBY", "node": lobby, "confidence": 0.1, "saran": []}

    kandidat.sort(key=lambda x: -x[0])
    best_skor, best_id, best_node = kandidat[0]
    conf  = round(min(best_skor, 1.0), 2)
    saran = [(c[1], c[2]["nama_ruangan"]) for c in kandidat[1:4] if c[2].get("nama_ruangan")]

    return {"found": conf >= 0.3, "node_id": best_id, "node": best_node,
            "confidence": conf, "saran": saran}


# ── Susun Langkah Navigasi Gabungan ──────────────────────────────────────────
def susun_navigasi(start_id: str, target_id: str, nodes: dict,
                   routes: dict, edges: dict) -> dict:
    """
    1. Jalankan A-Star untuk temukan jalur (node list + total jarak).
    2. Ambil langkah teks dari routes.csv (dari LOBBY ke tujuan).
    3. Jika start bukan LOBBY, sisipkan keterangan awal.
    """
    # A-Star
    astar_res = astar(start_id, target_id, edges, nodes)

    # Langkah teks dari routes.csv
    if target_id not in routes:
        return {"found": False,
                "error": f"Panduan arah ke '{target_id}' belum tersedia.",
                "astar": astar_res}

    route        = routes[target_id]
    langkah_teks = list(route["langkah_list"])

    # Prefix posisi awal jika bukan lobby
    if start_id != "LOBBY" and start_id in nodes:
        nd = nodes[start_id]
        langkah_teks.insert(0,
            f"Mulai dari {nd['nama_ruangan']} (Lt.{nd['lantai']} {nd['gedung']}), "
            f"kembali ke Lobby Utama terlebih dahulu")

    return {
        "found":         True,
        "langkah_teks":  langkah_teks,
        "catatan":       route.get("catatan_tambahan", ""),
        "jarak_rute":    route["total_jarak_estimasi"],
        "jarak_astar":   astar_res.get("total_jarak", 0),
        "path_astar":    astar_res.get("path", []),
        "lantai_tujuan": route["lantai_tujuan"],
        "nama_tujuan":   route["nama_tujuan"],
        "astar_found":   astar_res["found"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "history":         [],
    "start_node_id":   "LOBBY",
    "start_node":      nodes.get("LOBBY", {}),
    "start_confirmed": False,
    "start_input_val": "",
    "query_input":     "",
    "_last_query":     "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏫 CampusSeek")
    st.caption("Navigasi Ruangan Kampus Berbasis NLP")
    st.divider()

    # ── Posisi Awal ───────────────────────────────────────────────────────────
    st.markdown("**📍 Posisi Anda Sekarang**")
    st.caption("Tulis bebas — NLP akan mendeteksi otomatis")

    start_input = st.text_input(
        "Posisi awal",
        value=st.session_state.start_input_val,
        placeholder="contoh: lobby gedung A",
        label_visibility="collapsed",
        key="start_txt",
    )
    c1, c2 = st.columns([3,1])
    with c1:
        btn_set = st.button("📍 Tetapkan", use_container_width=True, type="primary")
    with c2:
        if st.button("↺", use_container_width=True, help="Reset"):
            for k in ("start_confirmed","start_input_val","_start_nlp"):
                st.session_state[k] = False if k=="start_confirmed" else ""
            st.session_state.start_node_id = "LOBBY"
            st.session_state.start_node    = nodes.get("LOBBY",{})
            st.rerun()

    if btn_set and start_input.strip():
        res = resolve_posisi_awal(start_input, nodes)
        st.session_state.start_input_val = start_input
        st.session_state.start_node_id   = res["node_id"]
        st.session_state.start_node      = res["node"]
        st.session_state.start_confirmed = res["found"]
        st.session_state["_start_nlp"]   = res
        st.rerun()

    # Feedback posisi
    nlp_start = st.session_state.get("_start_nlp", {})
    if st.session_state.start_confirmed and st.session_state.start_node:
        nd = st.session_state.start_node
        st.markdown(f"""<div class="pos-ok">
            ✅ <b>Terdeteksi:</b><br>
            📍 {nd['nama_ruangan']}<br>
            🏢 {nd['gedung']} · Lantai {nd['lantai']}<br>
            <small>Keyakinan NLP: {int(nlp_start.get('confidence',0)*100)}%</small>
        </div>""", unsafe_allow_html=True)
        saran = nlp_start.get("saran", [])
        if saran:
            st.caption("Bukan ini? Pilih:")
            for sid, snama in saran:
                if st.button(f"📍 {snama}", key=f"s_{sid}", use_container_width=True):
                    st.session_state.start_node_id = sid
                    st.session_state.start_node    = nodes[sid]
                    st.session_state.start_confirmed = True
                    st.rerun()
    elif st.session_state.start_input_val:
        st.markdown("""<div class="pos-warn">
            ⚠️ Posisi tidak dikenali.<br>
            Coba: <i>"lobby gedung A"</i>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Ketik posisi Anda lalu klik Tetapkan.")

    st.divider()

    # ── Contoh ────────────────────────────────────────────────────────────────
    st.markdown("**💡 Contoh Pertanyaan**")
    for c in ["Di mana lab basis data?","Ruang dekan ada di mana?",
              "Di mana perpustakaan?","Toilet di mana?",
              "Kaprodi kimia di mana?","Lab genetika lantai 2?",
              "Di mana aula?","Lab biokimia ada di mana?"]:
        if st.button(c, key=f"ex_{c}", use_container_width=True):
            st.session_state.query_input = c
            st.rerun()

    st.divider()
    st.markdown("**📊 Statistik**")
    ca, cb, cc = st.columns(3)
    ca.metric("Ruangan", len(nodes)-1)
    cb.metric("Rute",    len(routes))
    cc.metric("Intent",  len(intents))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="banner">
  <h1>🏫 CampusSeek</h1>
  <p>Sistem Navigasi Ruangan Kampus Berbasis Natural Language Processing</p>
  <p style="font-size:.8rem;opacity:.7;">
    IndoBERT · Klasifikasi Intent · Named Entity Recognition · Algoritma A-Star
  </p>
</div>""", unsafe_allow_html=True)

# Guard posisi awal
if not st.session_state.start_confirmed:
    st.warning("📍 **Tetapkan posisi awal Anda** di sidebar kiri sebelum mencari ruangan.")
    st.markdown("""**Contoh posisi yang bisa ditulis:**
- `lobby gedung A`
- `tangga lantai 2`
- `depan lab kimia`
- `koridor lantai 1`""")
    st.stop()

# ── Search Bar ────────────────────────────────────────────────────────────────
nd_start = st.session_state.start_node
st.markdown(
    f'<span class="nlp-badge b-start">📍 Dari: {nd_start["nama_ruangan"]} '
    f'(Lt.{nd_start["lantai"]})</span>',
    unsafe_allow_html=True
)

cq, cb2 = st.columns([5,1])
with cq:
    query = st.text_input("Cari ruangan",
        value=st.session_state.query_input,
        placeholder="Contoh: Di mana lab basis data? / Ruang dekan ada di mana?",
        label_visibility="collapsed", key="main_q")
with cb2:
    cari = st.button("🔍 Cari", use_container_width=True, type="primary")

# ── Proses ────────────────────────────────────────────────────────────────────
if query and (cari or query != st.session_state._last_query):
    st.session_state._last_query = query
    st.session_state.query_input = query

    with st.spinner("⚙️ Memproses NLP..."):
        entitas     = ner_extract(query)
        kandidat, intent_res = cari_node_tujuan(query, nodes, intents, entitas)

    # ── Panel NLP ─────────────────────────────────────────────────────────────
    with st.expander("🤖 Hasil Analisis NLP", expanded=True):
        st.markdown('<div class="nlp-panel">', unsafe_allow_html=True)
        st.markdown("**🏷️ Klasifikasi Intent**")
        st.markdown(
            f'<span class="nlp-badge b-intent">Intent: '
            f'{intent_res["intent_label"].replace("_"," ").title()}</span>'
            f'<span class="nlp-badge b-conf">Confidence: '
            f'{int(intent_res["confidence"]*100)}%</span>',
            unsafe_allow_html=True)
        st.progress(intent_res["confidence"],
                    text=f"Kecocokan: {int(intent_res['confidence']*100)}%")

        st.markdown("**🔍 Named Entity Recognition (NER)**")
        ner_badges = ""
        if entitas["lantai"]:
            ner_badges += f'<span class="nlp-badge b-lantai">Lantai: {entitas["lantai"]}</span>'
        if entitas["gedung"]:
            ner_badges += f'<span class="nlp-badge b-ner">Gedung: {entitas["gedung"]}</span>'
        if entitas["tipe_ruang"]:
            ner_badges += f'<span class="nlp-badge b-ner">Tipe: {entitas["tipe_ruang"].title()}</span>'
        if entitas["nama_prodi"]:
            ner_badges += f'<span class="nlp-badge b-ner">Prodi: {entitas["nama_prodi"].title()}</span>'
        if entitas["nama_khusus"]:
            ner_badges += f'<span class="nlp-badge b-ner">Entitas: {entitas["nama_khusus"].title()}</span>'
        if not ner_badges:
            ner_badges = '<span style="color:#888;font-size:.84rem;">Tidak ada entitas spesifik terdeteksi</span>'
        st.markdown(ner_badges, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if not kandidat:
        st.warning("⚠️ Ruangan tidak ditemukan. Coba kata kunci yang lebih spesifik.")
        st.stop()

    # Pilih kandidat
    target_score, target_id, target_node = kandidat[0]
    st.markdown("---")
    if len(kandidat) > 1:
        st.markdown(f"**Ditemukan {len(kandidat)} ruangan relevan:**")
        cols = st.columns(min(len(kandidat), 3))
        for i, (sc, nid, nd) in enumerate(kandidat[:3]):
            with cols[i]:
                lbl = f"{'✅ ' if i==0 else ''}{nd['nama_ruangan']}"
                if st.button(lbl, key=f"pick_{nid}", use_container_width=True):
                    target_id, target_node = nid, nd

    # Kartu ruangan tujuan
    st.markdown(f"""<div class="room-card">
        <h3>📍 {target_node['nama_ruangan']}</h3>
        <p>🏢 <b>Gedung</b>: {target_node['gedung']} &nbsp;|&nbsp;
           🔢 <b>Lantai</b>: {target_node['lantai']} &nbsp;|&nbsp;
           🏷️ <b>Kode</b>: {target_node['kode']} &nbsp;|&nbsp;
           📁 <b>Tipe</b>: {target_node['tipe'].title()}</p>
        <p style="margin-top:5px">📝 {target_node['deskripsi']}</p>
    </div>""", unsafe_allow_html=True)

    # ── Navigasi A-Star ───────────────────────────────────────────────────────
    with st.spinner("🗺️ Menghitung rute A-Star..."):
        nav = susun_navigasi(
            st.session_state.start_node_id,
            target_id, nodes, routes, edges
        )

    if not nav["found"]:
        st.error(f"❌ {nav.get('error','Rute tidak tersedia.')}")
        st.stop()

    # Metrik
    st.markdown(f"""<div class="metric-row">
        <div class="mc"><div class="val">📏 {nav['jarak_rute']} m</div>
          <div class="lbl">Jarak Estimasi</div></div>
        <div class="mc"><div class="val">🔢 {nav['jarak_astar']} m</div>
          <div class="lbl">Jarak A-Star</div></div>
        <div class="mc"><div class="val">🚶 {len(nav['langkah_teks'])}</div>
          <div class="lbl">Langkah</div></div>
        <div class="mc"><div class="val">⏱️ ~{max(1,nav['jarak_rute']//80)} mnt</div>
          <div class="lbl">Estimasi Waktu</div></div>
        <div class="mc"><div class="val">🏢 Lt.{nav['lantai_tujuan']}</div>
          <div class="lbl">Lantai Tujuan</div></div>
    </div>""", unsafe_allow_html=True)

    # A-Star path info
    if nav["astar_found"] and nav["path_astar"]:
        st.caption(f"🔵 Jalur A-Star: {' → '.join(nav['path_astar'])}")

    # Langkah navigasi
    st.markdown("### 🗺️ Petunjuk Arah")
    for i, step in enumerate(nav["langkah_teks"]):
        is_last = (i == len(nav["langkah_teks"]) - 1)
        cls = "step-box step-last" if is_last else "step-box"
        sl  = step.lower()
        icon = ("🪜" if "tangga" in sl
                else "✅" if any(x in sl for x in ["tiba","sampai","ada di"])
                else "↰"  if "belok kiri" in sl
                else "↱"  if "belok kanan" in sl
                else "⬆️")
        st.markdown(f"""<div class="{cls}">
            <div class="step-num">{i+1}</div>
            <div class="step-text">{icon} {step}</div>
        </div>""", unsafe_allow_html=True)

    if nav["catatan"].strip():
        st.markdown(
            f'<div class="note-box">📌 <b>Catatan:</b> {nav["catatan"]}</div>',
            unsafe_allow_html=True)

    # Simpan riwayat
    st.session_state.history.append({
        "query":   query,
        "dari":    nd_start["nama_ruangan"],
        "tujuan":  target_node["nama_ruangan"],
        "lantai":  nav["lantai_tujuan"],
        "jarak":   nav["jarak_rute"],
        "langkah": len(nav["langkah_teks"]),
    })

    st.markdown("---")
    st.caption("✨ CampusSeek | NLP Rule-based (upgrade ke IndoBERT) · A-Star Pathfinding")


# ══════════════════════════════════════════════════════════════════════════════
# TABS BAWAH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📋 Daftar Ruangan","🕘 Riwayat Pencarian","ℹ️ Panduan & Arsitektur NLP"])

with tab1:
    ca, cb, cc = st.columns(3)
    with ca:
        fg = st.selectbox("Gedung", ["Semua"] + sorted({n["gedung"] for n in nodes.values()}))
    with cb:
        fl = st.selectbox("Lantai", ["Semua","1","2","3"])
    with cc:
        ft = st.selectbox("Tipe", ["Semua"] + sorted({n["tipe"] for n in nodes.values()}))

    fil = [n for n in nodes.values() if n["node_id"] != "LOBBY"]
    if fg != "Semua": fil = [n for n in fil if n["gedung"] == fg]
    if fl != "Semua": fil = [n for n in fil if n["lantai"] == int(fl)]
    if ft != "Semua": fil = [n for n in fil if n["tipe"] == ft]

    st.caption(f"Menampilkan {len(fil)} ruangan")
    for n in fil:
        with st.expander(f"🏠 {n['nama_ruangan']} ({n['kode']}) — {n['gedung']} Lt.{n['lantai']}"):
            c1, c2 = st.columns(2)
            c1.write(f"**Tipe**: {n['tipe'].title()}")
            c1.write(f"**Gedung**: {n['gedung']}")
            c1.write(f"**Lantai**: {n['lantai']}")
            c2.write(f"**Kode**: {n['kode']}")
            c2.write(f"**Node ID**: {n['node_id']}")
            st.write(f"**Deskripsi**: {n['deskripsi']}")
            st.write(f"**Kata Kunci**: `{'`, `'.join(n['kata_kunci'])}`")
            if n["node_id"] in routes:
                if st.button(f"🔍 Navigasi ke sini", key=f"nav_{n['node_id']}"):
                    st.session_state.query_input = f"Di mana {n['nama_ruangan']}?"
                    st.rerun()

with tab2:
    if not st.session_state.history:
        st.info("Belum ada riwayat pencarian.")
    else:
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
        for h in reversed(st.session_state.history):
            st.markdown(f"""<div class="room-card">
                <h3>🔍 {h['query']}</h3>
                <p>📍 Dari: <b>{h['dari']}</b> &nbsp;|&nbsp;
                   ➡️ <b>{h['tujuan']}</b> &nbsp;|&nbsp;
                   🏢 Lt.{h['lantai']} &nbsp;|&nbsp;
                   📏 {h['jarak']} m &nbsp;|&nbsp;
                   🚶 {h['langkah']} langkah</p>
            </div>""", unsafe_allow_html=True)

with tab3:
    st.markdown("""
### 📖 Cara Menggunakan CampusSeek

**1. Tetapkan posisi awal (NLP)**
> Ketik posisi Anda di sidebar, misalnya: `lobby gedung A`, `tangga lantai 2`

**2. Cari ruangan tujuan**
> Tulis bebas: *"Di mana lab basis data?"*, *"Ruang dekan ada di mana?"*

**3. Ikuti petunjuk arah**
> Langkah navigasi ditampilkan beserta jarak A-Star dan estimasi waktu.

---
### 🏗️ Arsitektur NLP Sistem

```
Input Teks Pengguna
        │
        ├─ [NER] Named Entity Recognition
        │     ├─ Ekstrak Lantai  (regex: "lantai 2", "lt.1")
        │     ├─ Ekstrak Gedung  (regex: "gedung A")
        │     ├─ Ekstrak Tipe Ruang (keyword matching)
        │     └─ Ekstrak Nama Prodi / Entitas Khusus
        │
        ├─ [Intent] Klasifikasi Intent
        │     ├─ Token overlap vs intent_dataset.csv (Rule-based baseline)
        │     └─ → Upgrade: Fine-tuning IndoBERT pada dataset ini
        │
        └─ [Search] Pencarian Node Tujuan
              ├─ Skor kata_kunci + nama_ruangan + NER
              └─ Gabungkan dengan node_target dari Intent
                        │
                        ▼
              [A-Star] Pathfinding
              ├─ Heuristik: Jarak Euclidean (koordinat x,y nodes.csv)
              ├─ Graph: edges.csv (bidirectional)
              └─ Output: Path + Total Jarak + Langkah Teks (routes.csv)
```

---
### 📁 Struktur File

| File | Fungsi |
|------|--------|
| `nodes.csv` | Data ruangan + koordinat x,y untuk A-Star |
| `edges.csv` | Graf koneksi antar node + jarak meter |
| `routes.csv` | Panduan arah teks dari lobby ke tiap ruangan |
| `intent_dataset.csv` | 140 utterance berlabel untuk training intent classifier |


    """)
