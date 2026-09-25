"""Network Voronoi diagram via Multi-Source Dijkstra.

A network Voronoi diagram partitions the nodes of a graph into cells, one per
source node, where each cell contains the nodes closest to that source in
terms of shortest-path distance along the network edges.

This module uses networkx's ``multi_source_dijkstra`` (a single Dijkstra pass
seeded from all sources simultaneously) to compute, for every node, both its
nearest source and the distance to it, then visualizes the result with
matplotlib: node color encodes distance to the nearest source, and a
colorbar legend shows the distance scale.

It also computes each cell's shortest-path flow: for every node in a cell,
the shortest path to the cell's source is found, and every edge along such
a path counts one. Counts are stored as the edge attribute ``flow`` and
drawn as a tint of the source's color (plus an edge-width scale), so trunk
edges carrying many paths stand out against lightly used branches.

Run directly to see the four example networks drawn with their per-cell
shortest-path flow: a grid network, a random network (Erdos-Renyi), a
small-world network (Watts-Strogatz), and a dendritic planar network (tree).
"""

import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx


def network_voronoi(G, sources, weight="weight"):
    """Compute the network Voronoi partition of G given source nodes.

    Uses Multi-Source Dijkstra: a single Dijkstra traversal where all sources
    start with distance 0, so every node is reached first by its nearest
    source.

    Args:
        G: networkx graph (undirected or directed).
        sources: iterable of source node ids.
        weight: edge attribute used as distance weight.

    Returns:
        (dist, source_of): dist[node] is the shortest-path distance from
        node to its nearest source; source_of[node] is that nearest source.
    """
    sources = list(sources)
    if not sources:
        raise ValueError("at least one source node is required")
    missing = [s for s in sources if s not in G]
    if missing:
        raise ValueError(f"source nodes not in graph: {missing}")

    dist, paths = nx.multi_source_dijkstra(G, sources, weight=weight)
    source_of = {node: path[0] for node, path in paths.items()}
    return dist, source_of


def draw_network_voronoi(G, sources, pos=None, weight="weight", ax=None,
                         title=None, cmap="viridis", seed=42):
    """Draw a network Voronoi diagram on a matplotlib Axes.

    Nodes are drawn with a colorscale encoding distance to the nearest
    source (source nodes drawn as larger stars), and a colorbar legend
    shows the distance scale. Edges are drawn faintly for context.

    Args:
        G: networkx graph.
        sources: iterable of source node ids.
        pos: node layout dict (node -> (x, y)); computed with a spring
            layout if omitted.
        weight: edge attribute used as distance weight.
        ax: matplotlib Axes to draw on; created if omitted.
        title: optional plot title.
        cmap: matplotlib colormap name for the distance colorscale.
        seed: layout seed (used only when pos is None).

    Returns:
        The matplotlib Axes.
    """
    dist, source_of = network_voronoi(G, sources, weight=weight)

    if pos is None:
        # k raises the ideal spring length so nodes spread further apart.
        pos = nx.spring_layout(G, seed=seed, k=2.5 / (G.number_of_nodes() ** 0.5))
    if ax is None:
        _, ax = plt.subplots()

    sources = list(sources)
    source_set = set(sources)
    regular = [n for n in G.nodes if n not in source_set]

    # Assign each source a distinct color from tab10; every Voronoi region
    # (source + its edges) is drawn in that color.
    region_cmap = plt.get_cmap("tab10")
    region_color = {s: region_cmap(i % 10) for i, s in enumerate(sources)}

    # Edges colored by Voronoi region: an edge takes its endpoints' region
    # color when both endpoints agree (interior edge); edges between regions
    # (boundary edges) or with unreachable endpoints stay gray.
    interior = [e for e in G.edges
                if e[0] in source_of and e[1] in source_of
                and source_of[e[0]] == source_of[e[1]]]
    boundary = [e for e in G.edges if e not in set(interior)]
    nx.draw_networkx_edges(
        G, pos, edgelist=interior,
        edge_color=[region_color[source_of[e[0]]] for e in interior],
        width=2.0, alpha=0.8, ax=ax,
    )
    nx.draw_networkx_edges(G, pos, edgelist=boundary,
                           edge_color="gray", width=1.0, alpha=0.3, ax=ax)

    # Regular nodes: color = distance to nearest source; nodes unreachable
    # from every source (disconnected components) are drawn in light gray.
    reachable = [n for n in regular if n in dist]
    unreachable = [n for n in regular if n not in dist]
    node_collection = nx.draw_networkx_nodes(
        G, pos, nodelist=reachable,
        node_color=[dist[n] for n in reachable],
        cmap=cmap, node_size=80, ax=ax,
    )
    if unreachable:
        nx.draw_networkx_nodes(
            G, pos, nodelist=unreachable,
            node_color="lightgray", node_size=80, ax=ax,
        )

    # Source nodes: drawn on top as larger stars in their region color.
    nx.draw_networkx_nodes(
        G, pos, nodelist=sources,
        node_color=[region_color[s] for s in sources],
        node_shape="*", node_size=350,
        edgecolors="black", linewidths=1.0, ax=ax,
    )

    # Colorbar legend for the distance scale.
    cbar = plt.colorbar(node_collection, ax=ax)
    cbar.set_label("Distance to nearest source")

    # Mark each source with its id so the tab10 colors are identifiable.
    for s in sources:
        ax.annotate(str(s), pos[s], textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8,
                    fontweight="bold")

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def voronoi_edge_flow(G, sources, source_of=None, weight="weight",
                      flow="flow"):
    """Count how many node-to-source shortest paths land on each edge.

    For every Voronoi cell (a source plus the nodes assigned to it), each
    node's shortest path to the cell's source is computed inside the cell
    subgraph, and every edge along the path accumulates one count. The
    count is stored as the edge's ``flow`` attribute, so trunk edges near
    the source end up with large counts and leaf edges with 1.

    Without distance ties, a node's shortest path to its nearest source
    never leaves its cell, so the per-cell computation matches the global
    shortest path. With ties a cell can be pinched off; the path then falls
    back to the full graph.

    Args:
        G: networkx graph (undirected or directed; simple, not Multi).
        sources: iterable of source node ids.
        source_of: the Voronoi partition (node -> nearest source) as
            returned by ``network_voronoi``; recomputed if omitted.
        weight: edge attribute used as distance weight for shortest paths.
        flow: name of the edge attribute to store the count in.

    Returns:
        flow_of: dict mapping every edge (u, v) of G to the number of
            node-to-source shortest paths that use it.
    """
    sources = list(sources)
    if source_of is None:
        _, source_of = network_voronoi(G, sources, weight=weight)

    cells = {s: [] for s in sources}
    for node, s in source_of.items():
        cells[s].append(node)

    nx.set_edge_attributes(G, 0, flow)
    for s, members in cells.items():
        cell = G.subgraph(members)
        for v in members:
            if v == s:
                continue
            try:
                path = nx.shortest_path(cell, v, s, weight=weight)
            except nx.NetworkXNoPath:  # pinched off by a distance tie
                path = nx.shortest_path(G, v, s, weight=weight)
            for a, b in zip(path, path[1:]):
                G[a][b][flow] += 1
    return nx.get_edge_attributes(G, flow)


def _tint(color, t):
    """Blend a color toward white: t=0 gives white, t=1 gives the color."""
    r, g, b = color[:3]
    return ((1 - t) + t * r, (1 - t) + t * g, (1 - t) + t * b)


def draw_voronoi_edge_flow(G, sources, pos=None, weight="weight", flow="flow",
                           ax=None, title=None, seed=42):
    """Draw the shortest-path flow map of a network Voronoi diagram.

    Each interior edge is drawn tinted with its cell's source color: the
    deeper the tint (and the wider the edge), the more node-to-source
    shortest paths land on it. Tint is normalized per cell, so every cell's
    busiest edge carries the full source color. Boundary edges between
    cells are drawn faint gray, regular nodes take a light tint of their
    cell color, and sources are drawn as stars in their full cell color.

    Args:
        G: networkx graph.
        sources: iterable of source node ids.
        pos: node layout dict (node -> (x, y)); computed with a spring
            layout if omitted.
        weight: edge attribute used as distance weight.
        flow: edge attribute holding the shortest-path counts (set by
            ``voronoi_edge_flow``).
        ax: matplotlib Axes to draw on; created if omitted.
        title: optional plot title.
        seed: layout seed (used only when pos is None).

    Returns:
        The matplotlib Axes.
    """
    sources = list(sources)
    _, source_of = network_voronoi(G, sources, weight=weight)
    flow_of = voronoi_edge_flow(G, sources, source_of=source_of,
                                weight=weight, flow=flow)

    if pos is None:
        # k raises the ideal spring length so nodes spread further apart.
        pos = nx.spring_layout(G, seed=seed, k=2.5 / (G.number_of_nodes() ** 0.5))
    if ax is None:
        _, ax = plt.subplots()

    source_set = set(sources)
    region_cmap = plt.get_cmap("tab10")
    region_color = {s: region_cmap(i % 10) for i, s in enumerate(sources)}

    # Same interior/boundary split as draw_network_voronoi: an edge belongs
    # to a cell when both endpoints are in it.
    interior = [e for e in G.edges
                if e[0] in source_of and e[1] in source_of
                and source_of[e[0]] == source_of[e[1]]]
    boundary = [e for e in G.edges if e not in set(interior)]

    # Per-cell busiest edge, for normalizing each cell's tint scale.
    cell_max = {s: 0 for s in sources}
    for e in interior:
        s = source_of[e[0]]
        cell_max[s] = max(cell_max[s], flow_of.get(e, 0))

    edge_color, edge_width = [], []
    for e in interior:
        s = source_of[e[0]]
        t = flow_of.get(e, 0) / cell_max[s] if cell_max[s] else 0.0
        # Floor tint so zero-flow interior edges stay faintly visible.
        edge_color.append(_tint(region_color[s], 0.1 + 0.9 * t))
        edge_width.append(0.8 + 3.2 * t)
    nx.draw_networkx_edges(G, pos, edgelist=interior,
                           edge_color=edge_color, width=edge_width, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=boundary,
                           edge_color="gray", width=1.0, alpha=0.3, ax=ax)

    # Regular nodes: a light tint of their cell color; nodes unreachable
    # from every source (disconnected components) are drawn in light gray.
    regular = [n for n in G.nodes if n not in source_set]
    reachable = [n for n in regular if n in source_of]
    unreachable = [n for n in regular if n not in source_of]
    nx.draw_networkx_nodes(
        G, pos, nodelist=reachable,
        node_color=[_tint(region_color[source_of[n]], 0.3) for n in reachable],
        node_size=60, edgecolors="gray", linewidths=0.5, ax=ax,
    )
    if unreachable:
        nx.draw_networkx_nodes(
            G, pos, nodelist=unreachable,
            node_color="lightgray", node_size=60, ax=ax,
        )

    # Source nodes: drawn on top as larger stars in their region color.
    nx.draw_networkx_nodes(
        G, pos, nodelist=sources,
        node_color=[region_color[s] for s in sources],
        node_shape="*", node_size=350,
        edgecolors="black", linewidths=1.0, ax=ax,
    )

    # Legend: one proxy per source showing the cell color and the top of
    # its tint scale, plus the boundary edge style.
    handles = [
        Line2D([0], [0], color=region_color[s], lw=4, marker="*",
               markersize=14, markerfacecolor=region_color[s],
               markeredgecolor="black",
               label=f"source {s}: up to {cell_max[s]} paths/edge")
        for s in sources
    ]
    handles.append(Line2D([0], [0], color="gray", lw=1, alpha=0.3,
                          label="boundary edge"))
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              framealpha=0.9)

    # Mark each source with its id so the tab10 colors are identifiable.
    for s in sources:
        ax.annotate(str(s), pos[s], textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8,
                    fontweight="bold")

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def random_dendritic_tree(n, rng):
    """Random dendritic (tree-like, hence planar) graph on n nodes.

    Grown by a branching process: each node in turn sprouts a random number
    of children (0-3), producing the irregular branching shape typical of
    dendritic networks (river systems, vasculature, etc.).
    """
    G = nx.Graph()
    G.add_node(0)
    frontier = [0]
    node = 1
    while node < n:
        parent = frontier.pop(0)
        for _ in range(rng.randint(0, 3)):
            if node >= n:
                break
            G.add_edge(parent, node)
            frontier.append(node)
            node += 1
        if not frontier:  # everyone stopped branching; restart from a leaf
            frontier.append(node - 1)
    return G


def dendritic_layout(G, root=0):
    """Planar tidy-tree layout: x from DFS leaf order, y from depth.

    Guaranteed crossing-free for trees, and shows the branching structure.
    """
    pos = {}
    counter = [0]

    def place(node, parent, depth):
        children = [c for c in G.neighbors(node) if c != parent]
        if not children:
            x = counter[0]
            counter[0] += 1
        else:
            xs = [place(c, node, depth + 1) for c in children]
            x = sum(xs) / len(xs)
        pos[node] = (x, -depth)
        return x

    place(root, None, 0)
    return pos


def build_examples():
    """Build the four example networks used as test cases.

    Returns a list of (title, G, sources, pos) tuples, where pos is a node
    layout or None (draw functions then compute a spring layout).
    """
    # Example 1: grid network, sources spread across a 10x10 lattice.
    n = 10
    G_grid = nx.grid_2d_graph(n, n)
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    pos_grid = {i: (i % n, -(i // n)) for i in G_grid.nodes}  # grid-aligned
    sources_grid = [12, 47, 83]

    # Example 2: random network (Erdos-Renyi) with random edge weights.
    rng = random.Random(42)
    G_random = nx.erdos_renyi_graph(60, 0.08, seed=42)
    for u, v in G_random.edges:
        G_random[u][v]["weight"] = rng.uniform(0.5, 2.0)
    sources_random = rng.sample(list(G_random.nodes), 4)

    # Example 3: small-world network (Watts-Strogatz) with random weights.
    rng = random.Random(7)
    G_small_world = nx.watts_strogatz_graph(60, 4, 0.1, seed=7)
    for u, v in G_small_world.edges:
        G_small_world[u][v]["weight"] = rng.uniform(0.5, 2.0)
    sources_small_world = rng.sample(list(G_small_world.nodes), 4)

    # Example 4: random dendritic planar graph (a tree) with random weights.
    rng = random.Random(11)
    G_dendritic = random_dendritic_tree(60, rng)
    for u, v in G_dendritic.edges:
        G_dendritic[u][v]["weight"] = rng.uniform(0.5, 2.0)
    pos_dendritic = dendritic_layout(G_dendritic)
    sources_dendritic = rng.sample(list(G_dendritic.nodes), 4)

    return [
        (f"Grid network ({n}x{n})", G_grid, sources_grid, pos_grid),
        ("Random network (Erdos-Renyi)", G_random, sources_random, None),
        ("Small-world network (Watts-Strogatz)", G_small_world,
         sources_small_world, None),
        ("Dendritic planar network (tree)", G_dendritic,
         sources_dendritic, pos_dendritic),
    ]


def build_grid_supply_sweep(n=15, max_supplies=10, seed=5):
    """Build a series of grid cases with a growing set of supply nodes.

    All cases share the same nxn grid and the same seeded random sample of
    ``max_supplies`` source nodes; case k uses the first k of them (k runs
    from 2 to ``max_supplies``), so the Voronoi partition can be watched
    evolving as supply nodes are added one at a time.

    Returns a list of (title, G, sources, pos) tuples like
    ``build_examples``; every tuple shares the same graph object and the
    same grid-aligned layout.
    """
    G = nx.grid_2d_graph(n, n)
    G = nx.convert_node_labels_to_integers(G)
    pos = {i: (i % n, -(i // n)) for i in G.nodes}  # grid-aligned
    pool = random.Random(seed).sample(list(G.nodes), max_supplies)
    return [
        (f"Grid network ({n}x{n}), {k} supply nodes", G, pool[:k], pos)
        for k in range(2, max_supplies + 1)
    ]


def main():
    """Build the example networks and plot their shortest-path flow maps."""
    examples = build_examples()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, (title, G, sources, pos) in zip(axes.flat, examples):
        draw_voronoi_edge_flow(G, sources, pos=pos, ax=ax, title=title)
    fig.suptitle("Shortest-path flow to Voronoi sources (edge tint = number "
                 "of node-to-source shortest paths using the edge, "
                 "normalized per cell)", fontsize=14)
    fig.tight_layout()

    sweep = build_grid_supply_sweep()
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 16))
    for ax, (title, G, sources, pos) in zip(axes2.flat, sweep):
        draw_voronoi_edge_flow(G, sources, pos=pos, ax=ax, title=title)
    fig2.suptitle("15x15 grid with a growing set of supply nodes "
                  "(2 to 10, added one at a time)", fontsize=14)
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
