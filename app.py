"""
app.py — CampusSeek: Sistem Navigasi Ruangan Kampus Berbasis NLP
================================================================
Struktur CSV:
  nodes.csv   : node_id, nama_ruangan, kode, lantai, tipe, deskripsi, kata_kunci, gedung
  routes.csv  : route_id, titik_awal, ruangan_tujuan, node_tujuan, lantai_tujuan,
                langkah_navigasi, catatan_tambahan, gedung
  intents.csv : intent_id, intent_name, kategori, contoh_utterance_1..5,
                node_target, respon_template

Jalankan: streamlit run app.py
"""

import re
import csv
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CampusSeek",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

.banner {
    background: linear-gradient(135deg, #1B3A6B 0%, #2E5FA3 100%);
    padding: 1.4rem 2rem; border-radius: 14px;
    margin-bottom: 1.2rem; color: white;
}
.banner h1 { font-size: 2rem; margin: 0; font-weight: 700; }
.banner p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

.stTextInput > div > div > input {
    border-radius: 30px !important;
    border: 2px solid #2E5FA3 !important;
    padding: 0.65rem 1.2rem !important;
    font-size: 1rem !important;
}

.room-card {
    background: #f0f6ff; border: 1px solid #cce0ff;
    border-left: 5px solid #2E5FA3; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.room-card h3 { margin: 0 0 0.3rem; color: #1B3A6B; font-size: 1.1rem; }
.room-card p  { margin: 0; color: #444; font-size: 0.88rem; }

.step-box {
    display: flex; align-items: flex-start; gap: 0.8rem;
    background: white; border: 1px solid #e0e8f0;
    border-radius: 10px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.step-num {
    background: #2E5FA3; color: white; border-radius: 50%;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.82rem; flex-shrink: 0;
}
.step-last .step-num { background: #1A7A4A; }
.step-text { color: #222; font-size: 0.92rem; line-height: 1.4; padding-top: 3px; }

.nlp-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    margin-right: 6px; margin-bottom: 4px;
}
.badge-intent  { background: #e8f0fe; color: #1558d6; }
.badge-room    { background: #e6f4ea; color: #1A7A4A; }
.badge-lantai  { background: #fde8e8; color: #B00020; }
.badge-start   { background: #e0f7fa; color: #006064; }
.badge-conf    { background: #f3e8ff; color: #6200ea; }

.metric-row { display: flex; gap: 1rem; margin: 0.8rem 0; flex-wrap: wrap; }
.metric-card {
    background: white; border: 1px solid #dde8f8;
    border-radius: 10px; padding: 0.7rem 1.2rem;
    text-align: center; min-width: 120px; flex: 1;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .val { font-size: 1.4rem; font-weight: 700; color: #2E5FA3; }
.metric-card .lbl { font-size: 0.78rem; color: #666; margin-top: 2px; }

.start-ok {
    background: #e8f5e9; border: 1px solid #a5d6a7;
    border-left: 5px solid #2e7d32; border-radius: 10px;
    padding: 0.7rem 1rem; margin-top: 0.5rem;
    font-size: 0.88rem; color: #1b5e20;
}
.start-warn {
    background: #fff3e0; border: 1px solid #ffcc80;
    border-left: 5px solid #ef6c00; border-radius: 10px;
    padding: 0.7rem 1rem; margin-top: 0.5rem;
    font-size: 0.88rem; color: #bf360c;
}
.note-box {
    background: #fffde7; border: 1px solid #ffe082;
    border-radius: 8px; padding: 0.5rem 0.9rem;
    font-size: 0.83rem; color: #5f4200; margin-top: 6px;
}
.sidebar-title { font-size: 1rem; font-weight: 600; color: #1B3A6B; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER  (baca langsung dari CSV, tanpa src/)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_nodes(path="data/nodes.csv"):
    """
    Kembalikan dict { node_id(str) : { semua kolom } }
    kata_kunci dipecah jadi list.
    """
    nodes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["lantai"] = int(row["lantai"])
            row["kata_kunci"] = [k.strip() for k in row["kata_kunci"].split(",")]
            nodes[row["node_id"]] = row
    return nodes

@st.cache_data
def load_routes(path="data/routes.csv"):
    """
    Kembalikan dict { node_tujuan(str) : { semua kolom } }
    langkah_navigasi dipecah jadi list by " | ".
    """
    routes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["lantai_tujuan"] = int(row["lantai_tujuan"])
            row["langkah_list"] = [s.strip() for s in row["langkah_navigasi"].split("|") if s.strip()]
            routes[row["node_tujuan"]] = row
    return routes

@st.cache_data
def load_intents(path="data/intents.csv"):
    """
    Kembalikan list of dict, tiap dict = satu intent beserta utterance-nya.
    """
    intents = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            utterances = [
                row.get(f"contoh_utterance_{i}", "")
                for i in range(1, 6)
                if row.get(f"contoh_utterance_{i}", "").strip()
            ]
            row["utterances"] = utterances
            intents.append(row)
    return intents


# ═══════════════════════════════════════════════════════════════════════════════
# NLP PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def ekstrak_lantai(teks: str):
    """Ekstrak nomor lantai dari teks bebas."""
    m = re.search(r"(?:lantai|lt\.?)\s*(\d)", teks.lower())
    if m:
        return int(m.group(1))
    return None

def ekstrak_gedung(teks: str):
    """Ekstrak nama gedung dari teks bebas."""
    m = re.search(r"gedung\s*([a-zA-Z])", teks.lower())
    if m:
        return "Gedung " + m.group(1).upper()
    return None

def cari_ruangan(query: str, nodes: dict, lantai=None, gedung=None):
    """
    Cari node yang cocok dengan query berdasarkan:
    - nama_ruangan
    - kata_kunci
    - tipe
    Kembalikan list node diurutkan skor tertinggi.
    """
    query_lower = query.lower()
    kata_query  = set(re.findall(r"\w+", query_lower))
    hasil = []

    for nid, node in nodes.items():
        if nid == "LOBBY":
            continue  # Lobby bukan tujuan navigasi

        skor = 0.0
        nama_lower = node["nama_ruangan"].lower()
        tipe_lower = node["tipe"].lower()

        # Cocokkan kata_kunci
        for kw in node["kata_kunci"]:
            kw_lower = kw.lower()
            if kw_lower in query_lower:
                skor += 0.5
            elif any(k in kw_lower for k in kata_query if len(k) > 3):
                skor += 0.2

        # Cocokkan nama_ruangan
        for kata in kata_query:
            if len(kata) > 3 and kata in nama_lower:
                skor += 0.4

        # Cocokkan tipe
        if tipe_lower in query_lower:
            skor += 0.3

        # Filter lantai
        if lantai is not None:
            if node["lantai"] == lantai:
                skor += 0.3
            else:
                skor -= 0.3

        # Filter gedung
        if gedung:
            if gedung.lower() in node["gedung"].lower():
                skor += 0.2
            else:
                skor -= 0.2

        if skor > 0:
            hasil.append((skor, nid, node))

    hasil.sort(key=lambda x: -x[0])
    return [(nid, node, skor) for skor, nid, node in hasil]


def proses_query_tujuan(query: str, nodes: dict, intents: list):
    """
    Proses query tujuan pengguna.
    Kembalikan dict hasil NLP.
    """
    query_lower = query.lower()
    lantai  = ekstrak_lantai(query)
    gedung  = ekstrak_gedung(query)

    # ── Cocokkan intent dari dataset ──
    best_intent      = "cari_ruangan"
    best_intent_name = "Cari Ruangan"
    best_node_target = None
    best_conf        = 0.0

    for intent in intents:
        for utt in intent["utterances"]:
            utt_lower = utt.lower()
            # Hitung token overlap
            kata_utt   = set(re.findall(r"\w+", utt_lower))
            kata_query = set(re.findall(r"\w+", query_lower))
            if not kata_utt:
                continue
            overlap = len(kata_utt & kata_query) / len(kata_utt)
            if overlap > best_conf:
                best_conf        = overlap
                best_intent      = intent["intent_name"]
                best_intent_name = intent["intent_name"].replace("_", " ").title()
                best_node_target = intent["node_target"]

    # ── Cari kandidat ruangan ──
    kandidat = cari_ruangan(query, nodes, lantai, gedung)

    # Jika ada node_target dari intent dan ada di nodes → jadikan kandidat pertama
    if best_node_target and best_node_target in nodes and best_conf > 0.4:
        node_intent = nodes[best_node_target]
        # Sisipkan di depan jika belum ada
        existing_ids = [nid for nid, _, _ in kandidat]
        if best_node_target not in existing_ids:
            kandidat.insert(0, (best_node_target, node_intent, best_conf))

    return {
        "intent":     best_intent_name,
        "intent_raw": best_intent,
        "lantai":     lantai,
        "gedung":     gedung,
        "confidence": round(min(best_conf, 1.0), 2),
        "kandidat":   kandidat,   # list (node_id, node_dict, skor)
    }


def resolve_posisi_awal(teks: str, nodes: dict):
    """
    Proses teks bebas posisi awal → kembalikan node yang paling cocok.
    """
    if not teks.strip():
        return {"found": False, "node_id": "LOBBY", "node": nodes.get("LOBBY", {}), "confidence": 0.0, "saran": []}

    teks_lower = teks.lower()
    lantai  = ekstrak_lantai(teks)
    gedung  = ekstrak_gedung(teks)

    # Keyword tipe posisi
    tipe_kw = {
        "lobby":    ["lobby", "pintu masuk", "pintu depan", "entrance", "lobi", "masuk"],
        "tangga":   ["tangga", "stairs"],
        "koridor":  ["koridor", "lorong", "gang"],
        "toilet":   ["toilet", "wc", "kamar mandi"],
        "parkiran": ["parkir", "parkiran"],
    }
    tipe_target = None
    for tipe, kws in tipe_kw.items():
        if any(kw in teks_lower for kw in kws):
            tipe_target = tipe
            break

    kandidat = []
    for nid, node in nodes.items():
        skor = 0.0
        nama_lower = node["nama_ruangan"].lower()

        # Tipe cocok
        if tipe_target:
            if tipe_target in node["tipe"].lower():
                skor += 0.5
            elif tipe_target in nama_lower:
                skor += 0.35

        # Lantai cocok
        if lantai is not None:
            skor += 0.3 if node["lantai"] == lantai else -0.2
        elif node["lantai"] == 1:
            skor += 0.1   # default lantai 1

        # Gedung cocok
        if gedung:
            skor += 0.2 if gedung.lower() in node["gedung"].lower() else -0.1

        # Kata bebas cocok ke nama ruangan atau kata_kunci
        for kata in re.findall(r"\w+", teks_lower):
            if len(kata) > 3:
                if kata in nama_lower:
                    skor += 0.3
                if any(kata in kw.lower() for kw in node["kata_kunci"]):
                    skor += 0.2

        if skor > 0:
            kandidat.append((skor, nid, node))

    # Fallback ke LOBBY jika tidak ada kandidat
    if not kandidat:
        return {
            "found": False,
            "node_id": "LOBBY",
            "node": nodes.get("LOBBY", {}),
            "confidence": 0.1,
            "saran": []
        }

    kandidat.sort(key=lambda x: -x[0])
    best_skor, best_id, best_node = kandidat[0]
    conf = round(min(best_skor, 1.0), 2)
    saran = [(c[1], c[2]["nama_ruangan"]) for c in kandidat[1:4] if c[2].get("nama_ruangan")]

    return {
        "found":      conf >= 0.3,
        "node_id":    best_id,
        "node":       best_node,
        "confidence": conf,
        "saran":      saran,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGASI  (dari routes.csv langsung)
# ═══════════════════════════════════════════════════════════════════════════════

def get_navigasi(start_node_id: str, target_node_id: str, nodes: dict, routes: dict):
    """
    Ambil langkah navigasi dari routes.csv.
    Jika start bukan LOBBY, tambahkan langkah awal ke lobby dulu.
    Kembalikan dict { found, langkah_list, catatan, total_langkah }.
    """
    if target_node_id not in routes:
        return {"found": False, "error": f"Rute ke '{target_node_id}' belum tersedia di database."}

    route = routes[target_node_id]
    langkah = list(route["langkah_list"])  # copy

    # Jika posisi awal bukan lobby, sisipkan keterangan awal
    if start_node_id != "LOBBY" and start_node_id in nodes:
        nama_start = nodes[start_node_id]["nama_ruangan"]
        lantai_start = nodes[start_node_id]["lantai"]
        langkah.insert(0, f"Mulai dari {nama_start} (Lantai {lantai_start}), menuju Lobby Utama terlebih dahulu")

    return {
        "found":         True,
        "langkah_list":  langkah,
        "catatan":       route.get("catatan_tambahan", ""),
        "total_langkah": len(langkah),
        "lantai_tujuan": route["lantai_tujuan"],
        "ruangan_tujuan": route["ruangan_tujuan"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD SEMUA DATA
# ═══════════════════════════════════════════════════════════════════════════════
try:
    nodes   = load_nodes()
    routes  = load_routes()
    intents = load_intents()
except FileNotFoundError as e:
    st.error(f"❌ File CSV tidak ditemukan: {e}\n\nPastikan folder `data/` berisi nodes.csv, routes.csv, intents.csv")
    st.stop()


# ── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "history":         [],
    "start_node_id":   None,
    "start_node":      None,
    "start_confirmed": False,
    "start_input_val": "",
    "query_input":     "",
    "_last_query":     "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏫 CampusSeek")
    st.caption("Sistem Navigasi Ruangan Kampus Berbasis NLP")
    st.divider()

    # ── Posisi Awal (NLP Input) ───────────────────────────────────────────────
    st.markdown('<p class="sidebar-title">📍 Di mana posisi Anda sekarang?</p>', unsafe_allow_html=True)
    st.caption("Tulis bebas: *lobby*, *tangga lantai 2*, *depan toilet*, dll.")

    start_input = st.text_input(
        "Posisi awal",
        value=st.session_state.start_input_val,
        placeholder="contoh: lobby gedung A",
        label_visibility="collapsed",
        key="start_text_input",
    )

    col_ok, col_rst = st.columns([3, 1])
    with col_ok:
        btn_set = st.button("📍 Tetapkan Posisi", use_container_width=True, type="primary")
    with col_rst:
        btn_rst = st.button("↺", use_container_width=True, help="Reset posisi")

    if btn_rst:
        for k in ("start_node_id","start_node","start_confirmed","start_input_val","_start_nlp"):
            st.session_state[k] = None if k != "start_confirmed" else False
            if k == "start_input_val": st.session_state[k] = ""
        st.rerun()

    if btn_set and start_input.strip():
        hasil_start = resolve_posisi_awal(start_input, nodes)
        st.session_state.start_input_val = start_input
        st.session_state.start_node_id   = hasil_start["node_id"]
        st.session_state.start_node      = hasil_start["node"]
        st.session_state.start_confirmed = hasil_start["found"]
        st.session_state["_start_nlp"]   = hasil_start
        st.rerun()

    # Feedback deteksi posisi
    if st.session_state.start_confirmed and st.session_state.start_node:
        nd   = st.session_state.start_node
        conf = st.session_state.get("_start_nlp", {}).get("confidence", 0)
        st.markdown(f"""
        <div class="start-ok">
            ✅ <b>Posisi terdeteksi:</b><br>
            📍 {nd['nama_ruangan']}<br>
            🏢 {nd['gedung']} &nbsp;|&nbsp; Lantai {nd['lantai']}<br>
            <span style="font-size:0.78rem;opacity:0.8;">Keyakinan: {int(conf*100)}%</span>
        </div>
        """, unsafe_allow_html=True)

        saran = st.session_state.get("_start_nlp", {}).get("saran", [])
        if saran:
            st.caption("Maksud lain? Pilih:")
            for s_id, s_nama in saran:
                if st.button(f"📍 {s_nama}", key=f"saran_{s_id}", use_container_width=True):
                    st.session_state.start_node_id   = s_id
                    st.session_state.start_node      = nodes[s_id]
                    st.session_state["_start_nlp"]["confidence"] = 0.9
                    st.rerun()

    elif st.session_state.start_input_val and not st.session_state.start_confirmed:
        st.markdown("""
        <div class="start-warn">
            ⚠️ Posisi tidak dikenali.<br>
            Coba: <i>"lobby gedung A"</i> atau <i>"tangga lantai 2"</i>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("💡 Ketik posisi Anda lalu klik Tetapkan Posisi.")

    st.divider()

    # ── Contoh pertanyaan ─────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-title">💡 Contoh Pertanyaan</p>', unsafe_allow_html=True)
    contoh_list = [
        "Di mana lab basis data?",
        "Ruang dekan ada di mana?",
        "Di mana perpustakaan?",
        "Toilet di mana?",
        "Ruang kaprodi kimia di mana?",
        "Lab biokimia ada di lantai berapa?",
        "Di mana aula?",
        "Ruang seminar kimia biologi?",
        "Lab genetika di mana?",
        "Ruang rapat senat di mana?",
    ]
    for c in contoh_list:
        if st.button(c, key=f"ex_{c}", use_container_width=True):
            st.session_state.query_input = c
            st.rerun()

    st.divider()

    # ── Statistik database ────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-title">📊 Database Kampus</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ruangan", len(nodes) - 1)   # minus LOBBY
    c2.metric("Rute", len(routes))
    c3.metric("Intent", len(intents))

    tipe_cnt = {}
    for nd in nodes.values():
        t = nd["tipe"]
        tipe_cnt[t] = tipe_cnt.get(t, 0) + 1
    for t, c in sorted(tipe_cnt.items(), key=lambda x: -x[1]):
        st.caption(f"• {t.title()}: {c}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
    <h1>🏫 CampusSeek</h1>
    <p>Sistem Navigasi Ruangan Kampus Berbasis Natural Language Processing</p>
    <p style="font-size:0.82rem;opacity:0.7;">
        Rule-based NLP · Klasifikasi Intent · Named Entity Recognition · Navigasi Berbasis Rute
    </p>
</div>
""", unsafe_allow_html=True)

# Guard posisi awal
if not st.session_state.start_confirmed:
    st.warning("📍 **Tetapkan posisi awal Anda** di sidebar kiri sebelum mencari ruangan.")
    st.markdown("""
    **Contoh yang bisa ditulis di kolom posisi awal:**
    - `lobby gedung A`
    - `tangga lantai 2`
    - `koridor lantai 1`
    - `depan toilet`
    """)
    st.stop()

# ── Search bar ────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Cari ruangan",
        value=st.session_state.query_input,
        placeholder="Contoh: Di mana lab basis data? / Ruang dekan ada di mana?",
        label_visibility="collapsed",
        key="main_input",
    )
with col_btn:
    cari = st.button("🔍 Cari", use_container_width=True, type="primary")

# Badge posisi aktif
nd_start = st.session_state.start_node
if nd_start:
    st.markdown(
        f'<span class="nlp-badge badge-start">📍 Dari: {nd_start["nama_ruangan"]} '
        f'(Lt.{nd_start["lantai"]} {nd_start["gedung"]})</span>',
        unsafe_allow_html=True
    )

# ── Proses query ──────────────────────────────────────────────────────────────
if query and (cari or query != st.session_state._last_query):
    st.session_state._last_query  = query
    st.session_state.query_input  = query

    with st.spinner("Memproses query NLP..."):
        nlp = proses_query_tujuan(query, nodes, intents)

    # Tampilkan badge NLP
    with st.expander("🤖 Hasil Analisis NLP", expanded=True):
        badges = (
            f'<span class="nlp-badge badge-intent">Intent: {nlp["intent"]}</span>'
            + (f'<span class="nlp-badge badge-lantai">Lantai: {nlp["lantai"]}</span>' if nlp["lantai"] else "")
            + (f'<span class="nlp-badge badge-room">Gedung: {nlp["gedung"]}</span>'  if nlp["gedung"]  else "")
            + f'<span class="nlp-badge badge-conf">Confidence: {int(nlp["confidence"]*100)}%</span>'
        )
        st.markdown(badges, unsafe_allow_html=True)
        st.progress(nlp["confidence"], text=f"Kecocokan intent: {int(nlp['confidence']*100)}%")

    kandidat = nlp["kandidat"]

    if not kandidat:
        st.warning("⚠️ Ruangan tidak ditemukan. Coba gunakan kata kunci yang lebih spesifik.")
        st.stop()

    st.markdown("---")

    # Pilih ruangan jika ada lebih dari 1 kandidat
    target_id, target_node, _ = kandidat[0]
    if len(kandidat) > 1:
        st.markdown(f"**Ditemukan {len(kandidat)} ruangan relevan — menampilkan yang paling sesuai:**")
        cols = st.columns(min(len(kandidat), 3))
        for i, (nid, nd, sk) in enumerate(kandidat[:3]):
            with cols[i]:
                label = f"{'✅ ' if i==0 else ''}{nd['nama_ruangan']}"
                if st.button(label, key=f"pick_{nid}", use_container_width=True):
                    target_id   = nid
                    target_node = nd

    # Kartu ruangan tujuan
    st.markdown(f"""
    <div class="room-card">
        <h3>📍 {target_node['nama_ruangan']}</h3>
        <p>
            🏢 <b>Gedung</b>: {target_node['gedung']} &nbsp;|&nbsp;
            🔢 <b>Lantai</b>: {target_node['lantai']} &nbsp;|&nbsp;
            🏷️ <b>Kode</b>: {target_node['kode']} &nbsp;|&nbsp;
            📁 <b>Tipe</b>: {target_node['tipe'].title()}
        </p>
        <p style="margin-top:6px;">📝 {target_node['deskripsi']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigasi ──────────────────────────────────────────────────────────────
    with st.spinner("Mengambil rute navigasi..."):
        nav = get_navigasi(
            st.session_state.start_node_id,
            target_id,
            nodes,
            routes,
        )

    if not nav["found"]:
        st.error(f"❌ {nav['error']}")
        st.stop()

    # Metrik ringkasan
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="val">🚶 {nav['total_langkah']}</div>
            <div class="lbl">Jumlah Langkah</div>
        </div>
        <div class="metric-card">
            <div class="val">🏢 Lt. {nav['lantai_tujuan']}</div>
            <div class="lbl">Lantai Tujuan</div>
        </div>
        <div class="metric-card">
            <div class="val">📍 {nd_start['lantai']}</div>
            <div class="lbl">Lantai Awal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Langkah navigasi
    st.markdown("### 🗺️ Petunjuk Arah")
    steps = nav["langkah_list"]
    for i, step in enumerate(steps):
        is_last = (i == len(steps) - 1)
        cls     = "step-box step-last" if is_last else "step-box"
        sl      = step.lower()
        icon = ("🪜" if "tangga" in sl
                else "✅" if ("tiba" in sl or "sampai" in sl)
                else "↰" if "belok kiri" in sl
                else "↱" if "belok kanan" in sl
                else "⬆️")
        st.markdown(f"""
        <div class="{cls}">
            <div class="step-num">{i+1}</div>
            <div class="step-text">{icon} {step}</div>
        </div>
        """, unsafe_allow_html=True)

    # Catatan tambahan
    if nav["catatan"].strip():
        st.markdown(f'<div class="note-box">📌 <b>Catatan:</b> {nav["catatan"]}</div>', unsafe_allow_html=True)

    # Simpan riwayat
    st.session_state.history.append({
        "query":   query,
        "dari":    nd_start["nama_ruangan"],
        "tujuan":  target_node["nama_ruangan"],
        "lantai":  nav["lantai_tujuan"],
        "langkah": nav["total_langkah"],
    })

    st.markdown("---")
    st.caption("✨ CampusSeek — NLP Rule-based | Data: nodes.csv · routes.csv · intents.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS BAWAH
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📋 Daftar Ruangan", "🕘 Riwayat Pencarian", "ℹ️ Panduan"])

# Tab 1 — Daftar ruangan
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        gedung_list = sorted(set(nd["gedung"] for nd in nodes.values()))
        f_gedung = st.selectbox("Filter Gedung", ["Semua"] + gedung_list)
    with c2:
        f_lantai = st.selectbox("Filter Lantai", ["Semua", "1", "2", "3"])
    with c3:
        tipe_list = sorted(set(nd["tipe"] for nd in nodes.values()))
        f_tipe = st.selectbox("Filter Tipe", ["Semua"] + tipe_list)

    filtered = [nd for nd in nodes.values() if nd["node_id"] != "LOBBY"]
    if f_gedung != "Semua":
        filtered = [nd for nd in filtered if nd["gedung"] == f_gedung]
    if f_lantai != "Semua":
        filtered = [nd for nd in filtered if nd["lantai"] == int(f_lantai)]
    if f_tipe != "Semua":
        filtered = [nd for nd in filtered if nd["tipe"] == f_tipe]

    st.caption(f"Menampilkan {len(filtered)} ruangan")
    for nd in filtered:
        with st.expander(f"🏠 {nd['nama_ruangan']} ({nd['kode']}) — {nd['gedung']} Lt.{nd['lantai']}"):
            ca, cb = st.columns(2)
            ca.write(f"**Tipe**: {nd['tipe'].title()}")
            ca.write(f"**Gedung**: {nd['gedung']}")
            ca.write(f"**Lantai**: {nd['lantai']}")
            cb.write(f"**Kode**: {nd['kode']}")
            cb.write(f"**Node ID**: {nd['node_id']}")
            st.write(f"**Deskripsi**: {nd['deskripsi']}")
            st.write(f"**Kata Kunci**: `{'`, `'.join(nd['kata_kunci'])}`")
            if nd["node_id"] in routes:
                if st.button(f"🔍 Navigasi ke sini", key=f"nav_{nd['node_id']}"):
                    st.session_state.query_input = f"Di mana {nd['nama_ruangan']}?"
                    st.rerun()

# Tab 2 — Riwayat
with tab2:
    if not st.session_state.history:
        st.info("Belum ada riwayat pencarian.")
    else:
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
        for h in reversed(st.session_state.history):
            st.markdown(f"""
            <div class="room-card">
                <h3>🔍 {h['query']}</h3>
                <p>
                    📍 Dari: <b>{h['dari']}</b> &nbsp;|&nbsp;
                    ➡️ Tujuan: <b>{h['tujuan']}</b> &nbsp;|&nbsp;
                    🏢 Lt.{h['lantai']} &nbsp;|&nbsp;
                    🚶 {h['langkah']} langkah
                </p>
            </div>
            """, unsafe_allow_html=True)

# Tab 3 — Panduan
with tab3:
    st.markdown("""
    ### 📖 Cara Menggunakan CampusSeek

    **1. Tetapkan posisi awal (NLP)**
    > Ketik posisi Anda di sidebar, misalnya:
    > - `lobby gedung A`
    > - `tangga lantai 2`
    > - `koridor lantai 1`

    **2. Cari ruangan tujuan**
    > Tulis bebas seperti berbicara, contoh:
    > - *"Di mana lab basis data?"*
    > - *"Ruang dekan ada di mana?"*
    > - *"Lab genetika lantai 2 di mana?"*

    **3. Ikuti langkah navigasi**
    > Sistem menampilkan petunjuk arah langkah per langkah beserta catatan penanda khusus.

    ---
    ### 🗂️ Struktur File CSV

    | File | Kolom Utama |
    |------|------------|
    | `nodes.csv` | node_id, nama_ruangan, kode, lantai, tipe, deskripsi, kata_kunci, gedung |
    | `routes.csv` | route_id, titik_awal, ruangan_tujuan, node_tujuan, lantai_tujuan, langkah_navigasi, catatan_tambahan |
    | `intents.csv` | intent_id, intent_name, contoh_utterance_1..5, node_target, respon_template |

    ---
    ### 🤖 Alur NLP

    ```
    Input teks pengguna
         │
         ├─ Ekstrak lantai  (regex: "lantai 2", "lt.1")
         ├─ Ekstrak gedung  (regex: "gedung A")
         ├─ Cocokkan intent (token overlap vs utterance dataset)
         └─ Cari node       (skor kata_kunci + nama_ruangan + tipe)
                │
                └─ Ambil rute dari routes.csv → tampilkan langkah
    ```

    ---
    ### 📁 Struktur Folder
    ```
    campusseek/
    ├── data/
    │   ├── nodes.csv
    │   ├── routes.csv
    │   └── intents.csv
    └── app.py
    ```
    """)
