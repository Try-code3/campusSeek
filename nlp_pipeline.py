"""
nlp_pipeline.py
===============
NLP Pipeline untuk CampusSeek.

Tahap 1 (Rule-based) : berjalan tanpa GPU/model berat,
                       cocok untuk development & demo awal.

Tahap 2 (IndoBERT)   : aktifkan dengan install transformers + torch,
                       uncomment bagian IndoBERT di bawah.

Output:
    {
        "intent"         : str,   # label intent
        "entitas_ruang"  : str,
        "entitas_gedung" : str,
        "entitas_lantai" : int or None,
        "entitas_nama"   : str,
        "confidence"     : float
    }
"""

import re
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — RULE-BASED NLP (aktif sekarang, tanpa install tambahan)
# ═══════════════════════════════════════════════════════════════════════════════

# Kata kunci per intent
INTENT_KEYWORDS = {
    "cari_toilet": [
        "toilet", "wc", "kamar mandi", "buang air", "kakus", "lavatory"
    ],
    "cari_dosen": [
        "dosen", "pak ", "bu ", "bapak", "ibu", "kabin", "ruang dosen",
        "kantor dosen", "ketemu dosen"
    ],
    "cari_fasilitas": [
        "perpustakaan", "perpus", "library", "kopma", "koperasi", "kantin",
        "makan", "atm", "bank", "bem", "organisasi", "konseling", "bk",
        "bimbingan", "mushola", "sholat", "ibadah", "masjid", "tata usaha",
        "tu", "administrasi", "surat", "ruang baca"
    ],
    "tanya_posisi": [
        "saya di mana", "posisi saya", "saya ada di mana", "saya lagi di",
        "saya berada", "saya sekarang"
    ],
    "tanya_info": [
        "berapa lantai", "jam berapa", "buka sampai", "kapasitas", "muat berapa",
        "ada berapa gedung", "untuk apa"
    ],
    "cari_ruang": [
        "di mana", "ruang", "lab", "kelas", "aula", "sidang", "seminar",
        "mau ke", "cari", "tunjukkan", "cariin", "menuju", "sebelah mana",
        "lantai berapa", "gedung mana"
    ],
}

# Kata kunci gedung
GEDUNG_KEYWORDS = {
    "Gedung A": ["gedung a", "gedung-a", "gdg a"],
    "Gedung B": ["gedung b", "gedung-b", "gdg b"],
    "Gedung C": ["gedung c", "gedung-c", "gdg c", "perpustakaan", "perpus"],
}

# Kata kunci ruangan (dari rooms.csv kata_kunci utama)
ROOM_KEYWORDS = [
    ("lab jaringan",      ["lab jaringan", "jaringan", "networking", "cisco", "network"]),
    ("lab komputer",      ["lab komputer", "komputer dasar", "lab coding", "lab pemrograman"]),
    ("lab basis data",    ["lab basis data", "basis data", "database", "sql", "mysql"]),
    ("lab web",           ["lab web", "pemrograman web", "web", "html", "css", "javascript"]),
    ("lab multimedia",    ["lab multimedia", "multimedia", "desain grafis", "desain", "photoshop"]),
    ("aula",              ["aula", "serbaguna", "wisuda", "tempat acara"]),
    ("ruang sidang",      ["ruang sidang", "sidang", "ujian skripsi", "ruang ujian"]),
    ("ruang seminar",     ["ruang seminar", "seminar", "presentasi besar"]),
    ("perpustakaan",      ["perpustakaan", "perpus", "library", "pustaka"]),
    ("ruang baca",        ["ruang baca", "baca"]),
    ("koperasi",          ["koperasi", "kopma", "kantin", "makan"]),
    ("mushola",           ["mushola", "sholat", "ibadah", "masjid"]),
    ("tata usaha",        ["tata usaha", "tu", "administrasi", "surat", "berkas"]),
    ("atm",               ["atm", "mesin atm"]),
    ("ruang bem",         ["bem", "organisasi", "sekretariat", "himpunan"]),
    ("ruang konseling",   ["konseling", "bk", "bimbingan konseling", "konsultasi"]),
    ("ruang server",      ["server", "data center"]),
]

# Nama dosen yang dikenali
NAMA_KEYWORDS = [
    "ahmad", "pak ahmad", "ahmad fauzi",
    "sari", "bu sari", "sari wulandari",
    "budi", "pak budi", "budi santoso",
    "rina", "bu rina", "rina hastuti",
]


def detect_intent(text: str) -> tuple:
    """Deteksi intent dari teks. Kembalikan (intent, confidence)."""
    text_lower = text.lower()

    scores = {intent: 0 for intent in INTENT_KEYWORDS}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1

    best_intent = max(scores, key=scores.get)
    best_score  = scores[best_intent]

    if best_score == 0:
        return "cari_ruang", 0.5  # default

    # Normalise confidence 0.5 - 1.0
    confidence = min(0.5 + best_score * 0.15, 1.0)
    return best_intent, round(confidence, 2)


def extract_room(text: str) -> str:
    """Ekstrak entitas ruangan dari teks."""
    text_lower = text.lower()
    for room_name, keywords in ROOM_KEYWORDS:
        for kw in keywords:
            if kw in text_lower:
                return room_name
    # Coba ekstrak pola kode ruangan seperti A101, B202
    match = re.search(r'\b([a-cA-C][0-9]{3})\b', text)
    if match:
        return match.group(1).upper()
    return ""


def extract_gedung(text: str) -> str:
    """Ekstrak nama gedung dari teks."""
    text_lower = text.lower()
    for gedung_name, keywords in GEDUNG_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return gedung_name
    return ""


def extract_lantai(text: str) -> Optional[int]:
    """Ekstrak nomor lantai dari teks."""
    # Pola: "lantai 2", "lantai dua", "lt 3", "lt.2"
    match = re.search(r'lantai\s*(\d+)|lt\.?\s*(\d+)', text.lower())
    if match:
        val = match.group(1) or match.group(2)
        return int(val)
    # Kata angka
    angka_map = {"satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5}
    for kata, angka in angka_map.items():
        if f"lantai {kata}" in text.lower():
            return angka
    return None


def extract_nama(text: str) -> str:
    """Ekstrak nama orang (dosen) dari teks."""
    text_lower = text.lower()
    for nama in sorted(NAMA_KEYWORDS, key=len, reverse=True):
        if nama in text_lower:
            # Kembalikan format Title Case
            return nama.title()
    return ""


# ── Fungsi utama pipeline ─────────────────────────────────────────────────────

def process_query(text: str) -> Dict:
    """
    Proses teks pengguna → kembalikan dict hasil NLP.

    Parameters
    ----------
    text : str  — input teks bebas dari pengguna

    Returns
    -------
    {
        "intent"         : str,
        "entitas_ruang"  : str,
        "entitas_gedung" : str,
        "entitas_lantai" : int or None,
        "entitas_nama"   : str,
        "confidence"     : float,
        "mode"           : str  ("rule_based" atau "indobert")
    }
    """
    text = text.strip()
    if not text:
        return {
            "intent": "unknown",
            "entitas_ruang": "",
            "entitas_gedung": "",
            "entitas_lantai": None,
            "entitas_nama": "",
            "confidence": 0.0,
            "mode": "rule_based"
        }

    intent, confidence = detect_intent(text)
    ruang   = extract_room(text)
    gedung  = extract_gedung(text)
    lantai  = extract_lantai(text)
    nama    = extract_nama(text)

    # Jika intent dosen tapi belum ada entitas ruang → set ke "ruang dosen"
    if intent == "cari_dosen" and not ruang:
        ruang = "ruang dosen"

    # Jika intent toilet tapi belum ada entitas ruang → set ke "toilet"
    if intent == "cari_toilet" and not ruang:
        ruang = "toilet"

    return {
        "intent":          intent,
        "entitas_ruang":   ruang,
        "entitas_gedung":  gedung,
        "entitas_lantai":  lantai,
        "entitas_nama":    nama,
        "confidence":      confidence,
        "mode":            "rule_based"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — IndoBERT (uncomment setelah install: pip install transformers torch)
# ═══════════════════════════════════════════════════════════════════════════════
#
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# INTENT_LABELS = ["cari_ruang","cari_toilet","cari_dosen","cari_fasilitas",
#                  "tanya_posisi","tanya_info"]
#
# _tokenizer = None
# _model     = None
#
# def load_indobert_model(model_path: str = "./models/intent_model"):
#     global _tokenizer, _model
#     _tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
#     _model     = AutoModelForSequenceClassification.from_pretrained(model_path)
#     _model.eval()
#
# def predict_intent_indobert(text: str) -> tuple:
#     global _tokenizer, _model
#     if _tokenizer is None or _model is None:
#         load_indobert_model()
#     inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
#     with torch.no_grad():
#         logits = _model(**inputs).logits
#     probs     = torch.softmax(logits, dim=-1)[0]
#     best_idx  = probs.argmax().item()
#     return INTENT_LABELS[best_idx], round(probs[best_idx].item(), 2)
#
# Ganti baris detect_intent di process_query() dengan:
#     intent, confidence = predict_intent_indobert(text)
# ═══════════════════════════════════════════════════════════════════════════════


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Di mana lab jaringan lantai 2?",
        "Toilet wanita gedung B di mana?",
        "Ruang dosen Pak Budi ada di mana?",
        "Perpustakaan di mana?",
        "Mau ke aula gedung A",
        "Ruang sidang skripsi di mana?",
    ]
    for q in tests:
        r = process_query(q)
        print(f"\nQ : {q}")
        print(f"  intent   : {r['intent']} (conf={r['confidence']})")
        print(f"  ruang    : {r['entitas_ruang']}")
        print(f"  gedung   : {r['entitas_gedung']}")
        print(f"  lantai   : {r['entitas_lantai']}")
        print(f"  nama     : {r['entitas_nama']}")
