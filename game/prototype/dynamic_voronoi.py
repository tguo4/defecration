"""Dynamic Voronoi dismantling: greedy edge-removal animation.

Built on ``network_voronoi``: starting from the weighted Voronoi diagram
(each edge weighted by the number of node-to-source shortest paths using
it, as computed by ``network_voronoi.voronoi_edge_flow``), this module
repeatedly removes, from the current source's Voronoi cell, the edge with
the most shortest paths (largest flow weight), then recomputes the
diagram. Sources are processed in the order given; when a source's cell
subgraph has no edges left, dismantling moves on to the next source.

Frame 0 shows the untouched diagram; each subsequent frame removes exactly
one edge. Removed edges are drawn back in as faint red dashes so the
dismantling history stays visible.

Run directly to render one GIF animation per example network (grid,
Erdos-Renyi, small-world, dendritic) plus one per 15x15 grid supply-sweep
case (2 to 10 supply nodes) into the ``animations/`` directory.
"""

import os
import re

import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx

import network_voronoi as nv


def removal_sequence(G, sources, weight="weight"):
    """Simulate the greedy max-flow edge-removal process.

    Sources are dismantled in the order given. For the current source, the
    Voronoi partition and flow weights are recomputed on the current graph,
    the highest-flow edge inside that source's cell subgraph is removed
    (ties broken deterministically), and the process repeats until the
    cell subgraph has no edges left. Then the next source is dismantled.

    Args:
        G: networkx graph.
        sources: iterable of source node ids.
        weight: edge attribute used as distance weight.

    Returns:
        (frames, actions): frames[0] is the untouched graph copy, frames[i]
        the graph after i removals; actions[i] is None for frame 0 and a
        dict {edge, source, flow} describing the removal that produced
        frame i.
    """
    sources = list(sources)
    working = G.copy()
    frames = [working.copy()]
    actions = [None]

    for s in sources:
        while True:
            _, source_of = nv.network_voronoi(working, sources, weight=weight)
            cell_nodes = [n for n, src in source_of.items() if src == s]
            subgraph = working.subgraph(cell_nodes)
            if subgraph.number_of_edges() == 0:
                break
            flow = nv.voronoi_edge_flow(working, sources, source_of=source_of,
                                        weight=weight)
            # Highest-flow edge in the cell; ties broken by edge repr.
            edge = min(subgraph.edges(),
                       key=lambda e: (-_flow_on(flow, e), repr(e)))
            working.remove_edge(*edge)
            frames.append(working.copy())
            actions.append({"edge": edge, "source": s,
                            "flow": _flow_on(flow, edge)})
    return frames, actions


def _flow_on(flow, e):
    """Flow value for edge e, tolerating either endpoint order."""
    return flow.get(e, flow.get((e[1], e[0]), 0))


def animate_edge_removal(G, sources, pos=None, weight="weight",
                         interval=350, title=None, repeat=True):
    """Animate the greedy edge-removal dismantling of a Voronoi diagram.

    Each frame renders the current graph snapshot with
    ``network_voronoi.draw_voronoi_edge_flow`` (edge tint/width encode
    node-to-source shortest-path counts per cell). Edges removed in earlier
    steps are drawn as faint red dashes. The layout is computed once
    upfront so nodes stay fixed across frames.

    Args:
        G: networkx graph.
        sources: iterable of source node ids.
        pos: node layout dict; computed with a spring layout if omitted.
        weight: edge attribute used as distance weight.
        interval: milliseconds between frames.
        title: base title; the step/action line is appended per frame.
        repeat: whether the animation loops.

    Returns:
        (anim, frames, actions): matplotlib FuncAnimation plus the raw
        frame snapshots and action log from ``removal_sequence``.
    """
    frames, actions = removal_sequence(G, sources, weight=weight)
    if pos is None:
        # k raises the ideal spring length so nodes spread further apart.
        pos = nx.spring_layout(G, seed=42,
                               k=2.5 / (G.number_of_nodes() ** 0.5))

    # Wider than tall so the per-source legend fits outside the axes on
    # the right; constrained layout reserves the room every frame.
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    header = title or "Dynamic Voronoi"

    def update(i):
        ax.clear()
        snapshot = frames[i]
        step = f"step {i}/{len(frames) - 1}"
        if i == 0 or actions[i] is None:
            step += " — initial diagram"
        else:
            a = actions[i]
            step += (f" — removed edge {a['edge']} from cell of source "
                     f"{a['source']} (flow {a['flow']})")
        nv.draw_voronoi_edge_flow(snapshot, sources, pos=pos,
                                  weight=weight, ax=ax,
                                  title=f"{header}\n{step}")
        # Dismantling history: edges already removed, drawn back faintly.
        removed = [e for e in G.edges() if not snapshot.has_edge(*e)]
        if removed:
            nx.draw_networkx_edges(snapshot, pos, edgelist=removed,
                                   edge_color="red", style="dashed",
                                   width=1.0, alpha=0.35, ax=ax)

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=interval, repeat=repeat)
    return anim, frames, actions


def _slug(text):
    """Filesystem-friendly slug for a title string."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_example_animations(out_dir="animations", interval=350, dpi=90,
                            examples=None):
    """Render one animation per example network and save as GIF.

    Covers the four ``network_voronoi.build_examples`` networks plus every
    case of the 15x15 grid supply sweep (2 to 10 supply nodes), one GIF
    file per case. Pass ``examples`` to render only a subset, e.g.
    ``network_voronoi.build_grid_supply_sweep()``. Returns the list of
    written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if examples is None:
        examples = nv.build_examples() + nv.build_grid_supply_sweep()
    for title, G, sources, pos in examples:
        anim, frames, _ = animate_edge_removal(
            G, sources, pos=pos, interval=interval, title=title)
        path = os.path.join(out_dir, f"{_slug(title)}.gif")
        anim.save(path, writer=animation.PillowWriter(fps=1000 // interval),
                  dpi=dpi)
        written.append(path)
        print(f"saved {len(frames)} frames -> {path}")
    return written


def main():
    """Render one animation per example network."""
    save_example_animations()


if __name__ == "__main__":
    main()
