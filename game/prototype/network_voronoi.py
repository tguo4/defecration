"""Network Voronoi diagram via Multi-Source Dijkstra.

A network Voronoi diagram partitions the nodes of a graph into cells, one per
source node, where each cell contains the nodes closest to that source in
terms of shortest-path distance along the network edges.

This module uses networkx's ``multi_source_dijkstra`` (a single Dijkstra pass
seeded from all sources simultaneously) to compute, for every node, both its
nearest source and the distance to it, then visualizes the result with
matplotlib: node color encodes distance to the nearest source, and a
colorbar legend shows the distance scale.

Run directly to see three examples: a grid network, a random network
(Erdos-Renyi), and a small-world network (Watts-Strogatz).
"""

import random

import matplotlib.pyplot as plt
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


def main():
    """Build the example networks and plot them side by side."""
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flat
    draw_network_voronoi(G_grid, sources_grid, pos=pos_grid, ax=axes[0],
                         title=f"Grid network ({n}x{n}), sources: {sources_grid}")
    draw_network_voronoi(G_random, sources_random, ax=axes[1],
                         title=f"Random network (Erdos-Renyi), sources: {sources_random}")
    draw_network_voronoi(G_small_world, sources_small_world, ax=axes[2],
                         title=f"Small-world network (Watts-Strogatz), sources: {sources_small_world}")
    draw_network_voronoi(G_dendritic, sources_dendritic, pos=pos_dendritic, ax=axes[3],
                         title=f"Dendritic planar network (tree), sources: {sources_dendritic}")
    fig.suptitle("Network Voronoi diagrams (Multi-Source Dijkstra)", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
