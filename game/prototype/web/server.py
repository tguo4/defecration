"""Web frontend server for the network resilience demo.

Serves the interactive grid demo at http://localhost:8765 and exposes a
small JSON API built on the functions of ``network_resilience_demo.py``:

  POST /api/sample        {rows, cols, n_supplies, n_demands, seed}
                           -> sampled supply/demand node positions
  POST /api/perspectives  {rows, cols, supplies, demands, prefType,
                           prefValue}
                           -> one perspective per demand node

Run:  python server.py   then open http://localhost:8765
"""

import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network_resilience_demo import (  # noqa: E402
    perspective,
    prefer_k_nearest,
    prefer_within,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8765


def nstr(node):
    """Grid node (row, col) -> JSON-friendly "row,col" string."""
    return f"{node[0]},{node[1]}"


def ntup(s):
    r, c = s.split(",")
    return (int(r), int(c))


def sample_nodes(rows, cols, n_supplies, n_demands, seed):
    """Sample disjoint supply and demand node positions on the grid."""
    total = rows * cols
    if not (1 <= n_supplies + n_demands <= total):
        raise ValueError(
            f"supply + demand nodes ({n_supplies + n_demands}) must be "
            f"between 1 and the number of grid cells ({total})")
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    sampled = random.Random(seed).sample(nodes, n_supplies + n_demands)
    return sampled[:n_supplies], sampled[n_supplies:]


def compute_perspectives(rows, cols, supplies, demands, pref_type, pref_value):
    """One perspective per demand node, as a JSON-serializable dict."""
    G = nx.grid_2d_graph(rows, cols)
    supply_set = set(supplies)
    if pref_type == "rank":
        prefer = prefer_k_nearest(int(pref_value))
    elif pref_type == "distance":
        prefer = prefer_within(int(pref_value))
    else:
        raise ValueError(f"unknown preference type: {pref_type}")

    out = []
    for d in demands:
        if d not in G:
            raise ValueError(f"demand {nstr(d)} outside the {rows}x{cols} grid")
        p = perspective(G, d, supplies, prefer)
        edges = [{"a": nstr(a), "b": nstr(b), "flow": p.graph[a][b].get("flow", 0)}
                 for a, b in p.graph.edges]
        nodes = [nstr(v) for v in p.graph.nodes if v != d]
        out.append({
            "demand": nstr(d),
            "preferred": [nstr(s) for s in p.preferred],
            "distances": {nstr(s): p.distance[s] for s in p.preferred},
            "edges": edges,
            "nodes": nodes,
            "n_edges": p.graph.number_of_edges(),
        })
    # supplies outside every perspective would be unused info; not needed
    return out


class Handler(BaseHTTPRequestHandler):
    """Serves the page and the two JSON endpoints."""

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/sample":
                rows, cols = int(body["rows"]), int(body["cols"])
                if not (2 <= rows <= 60 and 2 <= cols <= 60):
                    raise ValueError("rows and cols must be between 2 and 60")
                supplies, demands = sample_nodes(
                    rows, cols, int(body["n_supplies"]),
                    int(body["n_demands"]), int(body["seed"]))
                self._send(200, {
                    "supplies": [nstr(s) for s in supplies],
                    "demands": [nstr(d) for d in demands],
                })
            elif self.path == "/api/perspectives":
                rows, cols = int(body["rows"]), int(body["cols"])
                supplies = [ntup(s) for s in body["supplies"]]
                demands = [ntup(d) for d in body["demands"]]
                unknown = [nstr(v) for v in supplies + demands
                            if not (0 <= v[0] < rows and 0 <= v[1] < cols)]
                if unknown:
                    raise ValueError(f"nodes outside the grid: {unknown}")
                self._send(200, {
                    "perspectives": compute_perspectives(
                        rows, cols, supplies, demands,
                        body["prefType"], int(body["prefValue"])),
                })
            else:
                self._send(404, {"error": "not found"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"bad request: {e}"})

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"network resilience demo: {url}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()
