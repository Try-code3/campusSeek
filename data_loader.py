"""
data_loader.py
==============
Load rooms.csv dan nodes.csv ke struktur data Python
yang siap dipakai oleh pathfinder dan nlp_pipeline.
"""

import csv
import os
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── Load Rooms ────────────────────────────────────────────────────────────────

def load_rooms(filepath: str = None) -> Dict[int, dict]:
    """
    Kembalikan dict:  room_id (int) → data ruangan (dict)
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "rooms.csv")

    rooms = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            room_id = int(row["room_id"])
            rooms[room_id] = {
                "room_id":    room_id,
                "nama_ruang": row["nama_ruang"].strip(),
                "kode_ruang": row["kode_ruang"].strip(),
                "gedung":     row["gedung"].strip(),
                "lantai":     int(row["lantai"]),
                "tipe_ruang": row["tipe_ruang"].strip(),
                "node_id":    int(row["node_id"]),
                "deskripsi":  row["deskripsi"].strip(),
                "kata_kunci": [k.strip().lower() for k in row["kata_kunci"].split(",")],
            }
    return rooms


def search_room_by_keyword(keyword: str, rooms: Dict[int, dict]) -> List[dict]:
    """
    Cari ruangan berdasarkan kata kunci (nama, kode, kata_kunci, tipe).
    Kembalikan list ruangan yang cocok, diurutkan berdasarkan relevansi.
    """
    keyword = keyword.lower().strip()
    results = []

    for room in rooms.values():
        score = 0

        # Exact match nama
        if keyword == room["nama_ruang"].lower():
            score += 10
        # Partial match nama
        elif keyword in room["nama_ruang"].lower():
            score += 7
        # Match kode
        elif keyword == room["kode_ruang"].lower():
            score += 8
        # Match tipe
        elif keyword in room["tipe_ruang"].lower():
            score += 4
        # Match kata kunci
        for kw in room["kata_kunci"]:
            if keyword in kw or kw in keyword:
                score += 3
                break

        if score > 0:
            results.append({**room, "_score": score})

    results.sort(key=lambda r: r["_score"], reverse=True)
    return results


def filter_rooms(rooms: Dict[int, dict],
                 gedung: str = None,
                 lantai: int = None,
                 tipe: str = None) -> List[dict]:
    """Filter ruangan berdasarkan gedung, lantai, dan/atau tipe."""
    result = []
    for room in rooms.values():
        if gedung and gedung.lower() not in room["gedung"].lower():
            continue
        if lantai and room["lantai"] != lantai:
            continue
        if tipe and tipe.lower() not in room["tipe_ruang"].lower():
            continue
        result.append(room)
    return result


# ── Load Nodes ────────────────────────────────────────────────────────────────

def load_nodes(filepath: str = None) -> Dict[int, dict]:
    """
    Kembalikan dict:  node_id (int) → data node (dict)
    Kolom 'tetangga' diparsing menjadi list of (node_id, jarak).
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "nodes.csv")

    nodes = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["node_id"])
            # Parse tetangga: "44:3.5|46:5.0" → [(44, 3.5), (46, 5.0)]
            tetangga = []
            raw = row["tetangga"].strip()
            if raw:
                for item in raw.split("|"):
                    item = item.strip()
                    if ":" in item:
                        parts = item.split(":")
                        try:
                            nb_id  = int(parts[0].strip())
                            nb_jarak = float(parts[1].strip())
                            tetangga.append((nb_id, nb_jarak))
                        except ValueError:
                            pass

            nodes[node_id] = {
                "node_id":       node_id,
                "x":             float(row["x"]),
                "y":             float(row["y"]),
                "lantai":        int(row["lantai"]),
                "gedung":        row["gedung"].strip(),
                "tipe_node":     row["tipe_node"].strip(),
                "nama_landmark": row["nama_landmark"].strip(),
                "tetangga":      tetangga,
            }
    return nodes


# ── Tangga / Lift Connector ───────────────────────────────────────────────────

def get_inter_building_edges() -> List[Tuple[int, int, float]]:
    """
    Koneksi antar gedung melalui jalan/koridor luar kampus.
    Format: (node_id_a, node_id_b, jarak_meter)
    """
    return [
        (1,  50, 30.0),   # Lobby Gedung A ↔ Lobby Gedung B
        (50, 90, 25.0),   # Lobby Gedung B ↔ Lobby Gedung C
        (1,  90, 40.0),   # Lobby Gedung A ↔ Lobby Gedung C
    ]


def get_stair_connections(nodes: Dict[int, dict]) -> List[Tuple[int, int]]:
    """
    Kembalikan list pasangan node tangga/lift yang berada di posisi
    x,y sama tapi lantai berbeda → koneksi antar lantai untuk A-Star.
    """
    stairs = [n for n in nodes.values() if n["tipe_node"] in ("tangga", "lift")]
    connections = []
    for i, a in enumerate(stairs):
        for b in stairs[i+1:]:
            if (a["gedung"] == b["gedung"] and
                    abs(a["x"] - b["x"]) < 0.5 and
                    abs(a["y"] - b["y"]) < 0.5 and
                    abs(a["lantai"] - b["lantai"]) == 1):
                connections.append((a["node_id"], b["node_id"]))
    return connections


if __name__ == "__main__":
    rooms = load_rooms()
    nodes = load_nodes()
    print(f"Rooms loaded : {len(rooms)}")
    print(f"Nodes loaded : {len(nodes)}")

    # Test pencarian
    hasil = search_room_by_keyword("lab jaringan", rooms)
    if hasil:
        print(f"\nHasil cari 'lab jaringan': {hasil[0]['nama_ruang']} "
              f"(node_id={hasil[0]['node_id']})")

    stairs = get_stair_connections(nodes)
    print(f"Stair connections: {stairs}")
