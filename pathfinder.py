"""
pathfinder.py
=============
Algoritma A-Star untuk navigasi indoor kampus.
Input  : node_start, node_goal, dict nodes
Output : list node path + teks petunjuk arah lengkap dengan jarak (meter)
"""

import math
import heapq
from typing import Dict, List, Optional, Tuple


# ── Heuristik Euclidean ───────────────────────────────────────────────────────

def heuristic(node_a: dict, node_b: dict) -> float:
    """
    Jarak Euclidean 2D antara dua node.
    Perbedaan lantai diberi penalti 5 meter per lantai.
    """
    dx = node_a["x"] - node_b["x"]
    dy = node_a["y"] - node_b["y"]
    dz = abs(node_a["lantai"] - node_b["lantai"]) * 5.0
    return math.sqrt(dx**2 + dy**2) + dz


# ── A-Star ────────────────────────────────────────────────────────────────────

def astar(start_id: int,
          goal_id: int,
          nodes: Dict[int, dict],
          extra_edges: List[Tuple[int, int]] = None) -> Optional[List[int]]:
    """
    Cari jalur terpendek dari start_id ke goal_id menggunakan A-Star.

    Parameters
    ----------
    start_id    : node_id titik awal
    goal_id     : node_id titik tujuan
    nodes       : dict dari data_loader.load_nodes()
    extra_edges : koneksi tambahan antar tangga/lift lintas lantai

    Returns
    -------
    List[int] urutan node_id dari start ke goal, atau None jika tidak ditemukan.
    """
    if start_id not in nodes or goal_id not in nodes:
        return None

    if start_id == goal_id:
        return [start_id]

    # Tambahkan edge tangga/lift (dua arah, jarak 4 meter per lantai)
    adjacency: Dict[int, List[Tuple[int, float]]] = {}
    for nid, node in nodes.items():
        adjacency[nid] = list(node["tetangga"])

    if extra_edges:
        for edge in extra_edges:
            if len(edge) == 3:
                a, b, dist = edge
            else:
                a, b = edge; dist = 4.0
            adjacency[a].append((b, dist))
            adjacency[b].append((a, dist))

    goal_node = nodes[goal_id]

    # Priority queue: (f_score, node_id)
    open_set: List[Tuple[float, int]] = []
    heapq.heappush(open_set, (0.0, start_id))

    came_from: Dict[int, int] = {}
    g_score: Dict[int, float] = {start_id: 0.0}
    f_score: Dict[int, float] = {start_id: heuristic(nodes[start_id], goal_node)}

    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_id:
            # Rekonstruksi path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_id)
            path.reverse()
            return path

        if current in visited:
            continue
        visited.add(current)

        for neighbor_id, edge_cost in adjacency.get(current, []):
            if neighbor_id not in nodes:
                continue
            tentative_g = g_score.get(current, float("inf")) + edge_cost

            if tentative_g < g_score.get(neighbor_id, float("inf")):
                came_from[neighbor_id] = current
                g_score[neighbor_id] = tentative_g
                f = tentative_g + heuristic(nodes[neighbor_id], goal_node)
                f_score[neighbor_id] = f
                heapq.heappush(open_set, (f, neighbor_id))

    return None  # Tidak ada jalur


# ── Kalkulasi total jarak ─────────────────────────────────────────────────────

def calculate_total_distance(path: List[int],
                              nodes: Dict[int, dict],
                              extra_edges: List[Tuple[int, int]] = None) -> float:
    """Hitung total jarak (meter) dari path A-Star."""
    total = 0.0
    edge_map = {}
    for nid, node in nodes.items():
        for nb_id, dist in node["tetangga"]:
            edge_map[(nid, nb_id)] = dist
            edge_map[(nb_id, nid)] = dist

    if extra_edges:
        for edge in extra_edges:
            if len(edge) == 3:
                a, b, dist = edge
            else:
                a, b = edge; dist = 4.0
            edge_map[(a, b)] = dist
            edge_map[(b, a)] = dist

    for i in range(len(path) - 1):
        key = (path[i], path[i+1])
        total += edge_map.get(key, 0.0)
    return round(total, 1)


# ── Instruction Generator ─────────────────────────────────────────────────────

def _get_direction(from_node: dict, to_node: dict) -> str:
    """Tentukan arah relatif berdasarkan perubahan koordinat x,y."""
    dx = to_node["x"] - from_node["x"]
    dy = to_node["y"] - from_node["y"]
    angle = math.degrees(math.atan2(dy, dx))

    if -30 <= angle <= 30:
        return "lurus"
    elif 30 < angle <= 150:
        return "belok kiri"
    elif angle > 150 or angle < -150:
        return "balik arah"
    else:
        return "belok kanan"


def generate_instructions(path: List[int],
                           nodes: Dict[int, dict],
                           nama_tujuan: str = "tujuan",
                           extra_edges: List[Tuple[int, int]] = None) -> List[str]:
    """
    Ubah list path node menjadi teks petunjuk arah step-by-step.

    Returns
    -------
    List[str] setiap elemen adalah satu langkah petunjuk arah.
    """
    if not path or len(path) < 2:
        return ["Anda sudah berada di lokasi tujuan."]

    instructions = []
    step = 1

    # Hitung edge map untuk jarak
    edge_map = {}
    for nid, node in nodes.items():
        for nb_id, dist in node["tetangga"]:
            edge_map[(nid, nb_id)] = dist
            edge_map[(nb_id, nid)] = dist
    if extra_edges:
        for edge in extra_edges:
            if len(edge) == 3:
                a, b, dist = edge
            else:
                a, b = edge; dist = 4.0
            edge_map[(a, b)] = dist
            edge_map[(b, a)] = dist

    i = 0
    while i < len(path) - 1:
        curr_node = nodes[path[i]]
        next_node = nodes[path[i + 1]]

        jarak = edge_map.get((path[i], path[i + 1]), 0.0)

        # Naik / turun tangga atau lift
        if (curr_node["tipe_node"] in ("tangga", "lift") and
                next_node["tipe_node"] in ("tangga", "lift") and
                curr_node["lantai"] != next_node["lantai"]):

            verb = "Naik" if next_node["lantai"] > curr_node["lantai"] else "Turun"
            transport = "tangga" if curr_node["tipe_node"] == "tangga" else "lift"
            instructions.append(
                f"Langkah {step}: {verb} {transport} menuju lantai {next_node['lantai']}."
            )
            step += 1
            i += 1
            continue

        # Akumulasi jarak lurus
        direction = _get_direction(curr_node, next_node)
        acc_jarak = jarak
        j = i + 1

        while j < len(path) - 1:
            next2 = nodes[path[j + 1]]
            next_dir = _get_direction(nodes[path[j]], next2)
            # Lantai berbeda → stop akumulasi
            if nodes[path[j]]["lantai"] != next2["lantai"]:
                break
            if next_dir == direction:
                acc_jarak += edge_map.get((path[j], path[j + 1]), 0.0)
                j += 1
            else:
                break

        acc_jarak = round(acc_jarak, 1)

        # Buat kalimat
        if direction == "lurus":
            landmark = next_node["nama_landmark"]
            if landmark and j < len(path) - 1:
                instructions.append(
                    f"Langkah {step}: Jalan lurus ± {acc_jarak} meter "
                    f"melewati {landmark}."
                )
            else:
                instructions.append(
                    f"Langkah {step}: Jalan lurus ± {acc_jarak} meter."
                )
        else:
            landmark = next_node["nama_landmark"]
            if landmark:
                instructions.append(
                    f"Langkah {step}: {direction.capitalize()} di {landmark}, "
                    f"lanjut ± {acc_jarak} meter."
                )
            else:
                instructions.append(
                    f"Langkah {step}: {direction.capitalize()}, "
                    f"lanjut ± {acc_jarak} meter."
                )

        step += 1
        i = j

    # Langkah terakhir: tiba
    last_node = nodes[path[-1]]
    instructions.append(
        f"Langkah {step}: Anda telah tiba di {nama_tujuan} "
        f"({last_node['nama_landmark'] or last_node['tipe_node']})."
    )

    return instructions


# ── Fungsi utama (dipanggil dari app.py) ─────────────────────────────────────

def find_route(start_node_id: int,
               goal_node_id: int,
               nodes: Dict[int, dict],
               nama_tujuan: str = "tujuan",
               extra_edges: List[Tuple[int, int]] = None
               ) -> dict:
    """
    Fungsi utama: cari rute dan kembalikan dict hasil lengkap.

    Returns
    -------
    {
        "found"        : bool,
        "path"         : List[int],
        "total_jarak"  : float,
        "instructions" : List[str],
        "error"        : str or None
    }
    """
    path = astar(start_node_id, goal_node_id, nodes, extra_edges)

    if path is None:
        return {
            "found": False,
            "path": [],
            "total_jarak": 0.0,
            "instructions": [],
            "error": "Rute tidak ditemukan. Periksa koneksi node di nodes.csv."
        }

    total = calculate_total_distance(path, nodes, extra_edges)
    instructions = generate_instructions(path, nodes, nama_tujuan, extra_edges)

    return {
        "found": True,
        "path": path,
        "total_jarak": total,
        "instructions": instructions,
        "error": None
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_nodes, get_stair_connections

    nodes = load_nodes()
    extra = get_stair_connections(nodes)

    # Test: Lobby Gedung A (node 1) → Lab Jaringan (node 65)
    result = find_route(1, 65, nodes, "Lab Jaringan", extra)
    print(f"Rute ditemukan : {result['found']}")
    print(f"Total jarak    : {result['total_jarak']} meter")
    print(f"Path nodes     : {result['path']}")
    print("\nPetunjuk Arah:")
    for ins in result["instructions"]:
        print(" ", ins)
