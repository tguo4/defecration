"""Network resilience demo: demand-node perspectives over a supply network.

A supply network moves flow from supply nodes (v+) to demand nodes (v-).
Each demand node depends on the network only through its own perspective:
the shortest paths connecting it to the supply nodes it prefers, where
preference is bounded by a per-demand preference function over travel
distance, such as "my 3 nearest supply nodes" (rank bound) or "every
supply node within 10 steps" (distance bound).

``perspective`` builds that subgraph: one Dijkstra pass from the demand
node measures the travel distance to every supply node, the preference
function selects the preferred supplies, and the union of the
demand-to-preferred-supply shortest paths is returned as a small graph.
Every edge of the union carries a ``flow`` attribute counting how many of
those paths share it (like ``network_voronoi.voronoi_edge_flow`` counts
node-to-source paths), so trunk edges whose loss would sever several of
the demand node's supplies at once stand out.

Run directly to see the demo network: a 15x15 grid with 15 supply nodes
and 4 demand nodes, drawn once as an overview and then once per
preference function, as one panel per demand node's perspective.
"""

import random
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx


Perspective = namedtuple("Perspective", ["graph", "demand", "preferred", "distance"])


def distances_to_supplies(G, demand, supplies, weight="weight"):
    """Travel distances from one demand node to every supply node.

    One Dijkstra pass from the demand node. Supplies the demand node
    cannot reach (disconnected components) are left out, so a preference
    function only ever chooses among reachable supplies.

    Args:
        G: networkx graph.
        demand: the demand node (v-) the distances are measured from.
        supplies: iterable of supply node ids (v+).
        weight: edge attribute used as distance weight.

    Returns:
        distances: dict mapping every reachable supply node to its
        shortest-path distance from the demand node.
    """
    length, _ = nx.single_source_dijkstra(G, demand, weight=weight)
    return {s: length[s] for s in supplies if s in length}


def prefer_k_nearest(k=3):
    """Preference function bounded by rank: the k nearest supply nodes.

    Returns a ``prefer(demand, distances)`` function keeping the k supply
    nodes with the smallest travel distance from the demand node, ties
    broken by supply id so the choice is deterministic.
    """
    def prefer(demand, distances):
        ranked = sorted(distances, key=lambda s: (distances[s], s))
        return set(ranked[:k])
    return prefer


def prefer_within(limit=10):
    """Preference function bounded by travel distance.

    Returns a ``prefer(demand, distances)`` function keeping every supply
    node within ``limit`` steps of the demand node.
    """
    def prefer(demand, distances):
        return {s for s, d in distances.items() if d <= limit}
    return prefer


def perspective(G, demand, supplies, prefer, weight="weight"):
    """One demand node's perspective: paths to its preferred supplies.

    The preference function bounds which supplies the demand node relies
    on (see ``prefer_k_nearest`` and ``prefer_within``). The perspective
    subgraph is the union of the shortest paths from the demand node to
    each preferred supply; each edge carries a ``flow`` attribute counting
    how many of those paths share it, so shared trunk edges (the weak
    points of the network for this demand node) can be spotted.

    Args:
        G: networkx graph (undirected or directed).
        demand: the demand node (v-) the perspective is taken from.
        supplies: iterable of supply node ids (v+).
        prefer: preference function ``prefer(demand, distances) -> set of
            supply nodes``, bounding the supplies the demand node relies on.
        weight: edge attribute used as distance weight.

    Returns:
        A ``Perspective(graph, demand, preferred, distance)`` named tuple:
        ``graph`` is the union-of-shortest-paths subgraph (edges keep their
        ``weight`` and gain a ``flow`` count), ``demand`` the demand node,
        ``preferred`` the sorted preferred supply ids, and ``distance`` each
        preferred supply's travel distance from the demand node.
    """
    supplies = list(supplies)
    missing = [v for v in [demand] + supplies if v not in G]
    if missing:
        raise ValueError(f"nodes not in graph: {missing}")

    distances = distances_to_supplies(G, demand, supplies, weight=weight)
    preferred = sorted(s for s in prefer(demand, distances) if s in distances)

    # Union of the shortest paths demand -> each preferred supply; ``flow``
    # counts the paths sharing each edge, so edges near the demand node
    # that several paths run along end up with the largest counts.
    H = nx.Graph()
    H.add_node(demand)
    for s in preferred:
        path = nx.shortest_path(G, demand, s, weight=weight)
        H.add_nodes_from(path)
        for a, b in zip(path, path[1:]):
            if not H.has_edge(a, b):
                H.add_edge(a, b, **G[a][b])
            H[a][b]["flow"] = H[a][b].get("flow", 0) + 1
    return Perspective(H, demand, tuple(preferred),
                       {s: distances[s] for s in preferred})


def _tint(color, t):
    """Blend a color toward white: t=0 gives white, t=1 gives the color."""
    r, g, b = color[:3]
    return ((1 - t) + t * r, (1 - t) + t * g, (1 - t) + t * b)


def demand_colors(demands):
    """tab10 color per demand node, assigned in the order given.

    Color follows the demand node, never its rank: a demand node keeps the
    same color in every figure of the demo.
    """
    cmap = plt.get_cmap("tab10")
    return {d: cmap(i % 10) for i, d in enumerate(demands)}


def draw_supply_demand_network(G, supplies, demands, pos=None, weight="weight",
                               ax=None, title=None, seed=42):
    """Draw the supply/demand network overview.

    Every edge is drawn faintly for context; plain nodes are small gray
    dots, supply nodes (v+) are gold stars, and demand nodes (v-) are
    triangles in their tab10 color (see ``demand_colors``). Node ids are
    annotated so the per-demand perspective panels can be cross-read.

    Args:
        G: networkx graph.
        supplies: iterable of supply node ids (v+).
        demands: iterable of demand node ids (v-).
        pos: node layout dict; computed with a spring layout if omitted.
        weight: unused here; kept for signature parity with the other
            drawing functions.
        ax: matplotlib Axes to draw on; created if omitted.
        title: optional plot title.
        seed: layout seed (used only when pos is None).

    Returns:
        The matplotlib Axes.
    """
    supplies, demands = list(supplies), list(demands)
    colors = demand_colors(demands)

    if pos is None:
        # k raises the ideal spring length so nodes spread further apart.
        pos = nx.spring_layout(G, seed=seed,
                               k=2.5 / (G.number_of_nodes() ** 0.5))
    if ax is None:
        _, ax = plt.subplots()

    special = set(supplies) | set(demands)
    regular = [n for n in G.nodes if n not in special]

    # Context: the whole network, faint.
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.0,
                            alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=regular,
                           node_color="lightgray", node_size=40, ax=ax)

    # Supply nodes (v+) as gold stars; demand nodes (v-) as triangles in
    # their own color, drawn on top.
    nx.draw_networkx_nodes(G, pos, nodelist=supplies, node_color="gold",
                           node_shape="*", node_size=350,
                           edgecolors="black", linewidths=1.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=demands,
                           node_color=[colors[d] for d in demands],
                           node_shape="^", node_size=450,
                           edgecolors="black", linewidths=1.0, ax=ax)

    # Mark every node with its id so the panels are cross-readable.
    for s in supplies:
        ax.annotate(str(s), pos[s], textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7)
    for d in demands:
        ax.annotate(str(d), pos[d], textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8,
                    fontweight="bold")

    handles = [
        Line2D([0], [0], marker="*", lw=0, color="gold",
               markeredgecolor="black", markersize=14,
               label="supply node (v+)"),
    ] + [
        Line2D([0], [0], marker="^", lw=0, color=colors[d],
               markeredgecolor="black", markersize=12,
               label=f"demand node (v-) {d}")
        for d in demands
    ] + [
        Line2D([0], [0], color="gray", lw=1, alpha=0.3,
               label="network edge"),
    ]
    # Legend sits outside the axes on the right so it never covers nodes.
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, framealpha=0.9, borderaxespad=0)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def draw_perspective(G, supplies, persp, color, pos=None, weight="weight",
                     ax=None, title=None, seed=42):
    """Draw one demand node's perspective over the faint full network.

    The perspective paths are drawn in the demand node's color: edge tint
    and width encode how many demand-to-preferred-supply shortest paths
    share the edge (the perspective ``flow``, normalized per perspective),
    like ``network_voronoi.draw_voronoi_edge_flow`` encodes cell flow.
    Preferred supply nodes are drawn as large gold stars annotated with
    their travel distance from the demand node; supplies left outside the
    preference stay small pale stars. The demand node itself is a triangle
    in its own color.

    Args:
        G: networkx graph the perspective lives in (drawn as context).
        supplies: iterable of all supply node ids (v+).
        persp: a ``Perspective`` as returned by ``perspective``.
        color: the demand node's color (see ``demand_colors``).
        pos: node layout dict; computed with a spring layout if omitted.
        ax: matplotlib Axes to draw on; created if omitted.
        title: optional plot title.
        seed: layout seed (used only when pos is None).

    Returns:
        The matplotlib Axes.
    """
    supplies = list(supplies)
    preferred = set(persp.preferred)
    H = persp.graph

    if pos is None:
        # k raises the ideal spring length so nodes spread further apart.
        pos = nx.spring_layout(G, seed=seed,
                               k=2.5 / (G.number_of_nodes() ** 0.5))
    if ax is None:
        _, ax = plt.subplots()

    supply_set = set(supplies)
    preferred_list = [s for s in supplies if s in preferred]
    other_list = [s for s in supplies if s not in preferred]

    # Context: the network minus the perspective paths, faint; nodes off
    # the perspective (and off the supply set) as small gray dots.
    path_edges = {frozenset(e) for e in H.edges}
    context_edges = [e for e in G.edges if frozenset(e) not in path_edges]
    context_nodes = [n for n in G.nodes
                     if n not in H and n not in supply_set]
    nx.draw_networkx_edges(G, pos, edgelist=context_edges,
                           edge_color="gray", width=1.0, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=context_nodes,
                           node_color="lightgray", node_size=40, ax=ax)

    # Perspective paths: tint and width encode how many of the demand's
    # shortest paths share each edge, normalized per perspective.
    max_flow = max((H[a][b].get("flow", 0) for a, b in H.edges), default=0)
    edge_color, edge_width = [], []
    for a, b in H.edges:
        t = H[a][b].get("flow", 0) / max_flow if max_flow else 0.0
        # Floor tint so single-path edges stay faintly visible.
        edge_color.append(_tint(color, 0.1 + 0.9 * t))
        edge_width.append(0.8 + 3.2 * t)
    nx.draw_networkx_edges(H, pos, edge_color=edge_color,
                           width=edge_width, ax=ax)

    # Nodes on the perspective paths (except the demand and supply nodes)
    # as small circles tinted with the demand color.
    path_nodes = [n for n in H.nodes
                  if n != persp.demand and n not in supply_set]
    nx.draw_networkx_nodes(H, pos, nodelist=path_nodes,
                           node_color=[_tint(color, 0.3) for _ in path_nodes],
                           node_size=60, edgecolors="gray",
                           linewidths=0.5, ax=ax)

    # Supplies: preferred ones as large gold stars, the rest pale, so the
    # bound set by the preference function is visible at a glance.
    nx.draw_networkx_nodes(G, pos, nodelist=other_list,
                           node_color="lightgray", node_shape="*",
                           node_size=200, edgecolors="gray",
                           linewidths=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=preferred_list,
                           node_color="gold", node_shape="*", node_size=400,
                           edgecolors="black", linewidths=1.0, ax=ax)

    # The demand node itself, on top, as a triangle in its own color.
    nx.draw_networkx_nodes(G, pos, nodelist=[persp.demand],
                           node_color=[color], node_shape="^", node_size=450,
                           edgecolors="black", linewidths=1.0, ax=ax)

    # Preferred supplies annotated with id and travel distance.
    for s in preferred_list:
        ax.annotate(f"{s} ({persp.distance[s]:g})", pos[s],
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=7, fontweight="bold")
    ax.annotate(str(persp.demand), pos[persp.demand],
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="^", lw=0, color=color,
               markeredgecolor="black", markersize=12,
               label=f"demand node (v-) {persp.demand}"),
        Line2D([0], [0], marker="*", lw=0, color="gold",
               markeredgecolor="black", markersize=14,
               label="preferred supply (v+)"),
        Line2D([0], [0], marker="*", lw=0, color="lightgray",
               markeredgecolor="gray", markersize=11,
               label="supply outside preference"),
        Line2D([0], [0], marker="o", lw=0,
               color=_tint(color, 0.3), markeredgecolor="gray",
               markersize=8, label="path node"),
        Line2D([0], [0], color=color, lw=4,
               label="perspective path (width = paths sharing the edge)"),
        Line2D([0], [0], color="gray", lw=1, alpha=0.3,
               label="network edge"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, framealpha=0.9, borderaxespad=0)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def build_grid_example(n=15, n_supplies=15, n_demands=4, seed=5):
    """Grid network with random supply (v+) and demand (v-) nodes.

    Args:
        n: grid side, so the network is nxn nodes.
        n_supplies: number of supply nodes to sample.
        n_demands: number of demand nodes to sample.
        seed: seed for sampling the supply and demand nodes.

    Returns:
        (G, supplies, demands, pos): the grid graph (integer node labels),
        the sampled supply ids, the sampled demand ids (disjoint from the
        supplies), and a grid-aligned layout.
    """
    G = nx.grid_2d_graph(n, n)
    G = nx.convert_node_labels_to_integers(G)
    pos = {i: (i % n, -(i // n)) for i in G.nodes}  # grid-aligned
    sampled = random.Random(seed).sample(list(G.nodes),
                                         n_supplies + n_demands)
    return G, sampled[:n_supplies], sampled[n_supplies:], pos


def main():
    """Render the grid overview and each demand node's perspective."""
    G, supplies, demands, pos = build_grid_example()
    colors = demand_colors(demands)

    # Overview: the whole supply/demand network.
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    draw_supply_demand_network(
        G, supplies, demands, pos=pos, ax=ax,
        title=f"Grid network: {len(supplies)} supply nodes (v+), "
              f"{len(demands)} demand nodes (v-)")

    # One figure per preference function: one panel per demand node.
    preferences = [
        ("3 nearest supply nodes", prefer_k_nearest(3)),
        ("supply nodes within 10 steps", prefer_within(10)),
    ]
    for label, prefer in preferences:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), layout="constrained")
        for ax, demand in zip(axes.flat, demands):
            persp = perspective(G, demand, supplies, prefer)
            draw_perspective(
                G, supplies, persp, colors[demand], pos=pos, ax=ax,
                title=f"demand {demand}: {len(persp.preferred)} preferred "
                      f"supplies, {persp.graph.number_of_edges()} path edges")
        fig.suptitle(f"Perspective of each demand node — preference: {label}",
                     fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()