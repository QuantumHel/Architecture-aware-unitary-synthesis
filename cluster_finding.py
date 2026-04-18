from collections import deque
from itertools import combinations
from math import inf

def cnot_count_for_uc_gates_per_recursion_level(recursion_level, n):
    cnot_count = 3 * (4 ** recursion_level) * (2 ** (n - recursion_level - 1))
    return cnot_count

def bfs_distances(source: int, neighbors: dict[int, set[int]]) -> dict[int, int]:
    """BFS shortest-path distances from a single source."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nb in neighbors.get(node, set()):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def all_pairs_distances(neighbors: dict[int, set[int]]) -> dict[int, dict[int, int]]:
    """Compute shortest-path distances between all pairs via repeated BFS."""
    return {node: bfs_distances(node, neighbors) for node in neighbors}


def cluster_cost(nodes: list[int], dist: dict[int, dict[int, int]]) -> float:
    """Sum of all pairwise shortest-path distances within a node set."""
    total = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = dist.get(nodes[i], {}).get(nodes[j], inf)
            if d == inf:
                return inf
            total += d
    return total


def order_cluster(
    cluster: list[int], dist: dict[int, dict[int, int]]
) -> list[int]:
    """
    Re-index a cluster so that:
      - Index 0 is the most central node (smallest total distance to
        every other node in the cluster).
      - Each subsequent index is the node closest to the already-ordered
        set (smallest sum of distances to all previously placed nodes),
        with ties broken by smallest node id.
    """
    remaining = set(cluster)

    # Index 0: node with the smallest total distance to all others
    def total_dist(node):
        return sum(dist[node].get(other, inf) for other in cluster if other != node)

    center = min(remaining, key=lambda n: (total_dist(n), n))
    ordered = [center]
    remaining.remove(center)

    # Each next index: closest to the already-ordered set
    while remaining:
        def cost_to_ordered(node):
            return sum(dist[node].get(o, inf) for o in ordered)

        nxt = min(remaining, key=lambda n: (cost_to_ordered(n), n))
        ordered.append(nxt)
        remaining.remove(nxt)

    return ordered


# ── Exact solver (brute-force, feasible for small graphs) ─────────

def find_closest_cluster_exact(
    n: int,
    neighbors: dict[int, set[int]],
) -> tuple[list[int], float]:
    """
    Try every combination of n nodes; return the one with the
    smallest sum of pairwise distances.

    Complexity: O(V^n) — only practical when C(|V|, n) is small.
    """
    dist = all_pairs_distances(neighbors)
    all_nodes = list(neighbors.keys())

    if n > len(all_nodes):
        raise ValueError(f"Requested cluster size {n} exceeds graph size {len(all_nodes)}")

    best_cluster = None
    best_cost = inf

    for combo in combinations(all_nodes, n):
        cost = cluster_cost(list(combo), dist)
        if cost < best_cost:
            best_cost = cost
            best_cluster = list(combo)

    return order_cluster(best_cluster, dist), dist


# ── Greedy solver (scalable heuristic) ────────────────────────────

def find_closest_cluster_greedy(
    n: int,
    neighbors: dict[int, set[int]],
) -> tuple[list[int], float]:
    """
    Greedy heuristic:
      1. Seed with the edge whose endpoints have the best
         average closeness to the rest of the graph.
      2. Repeatedly add the node that increases the total
         pairwise distance of the cluster the least.

    Complexity: O(V^2) for BFS + O(n * V) for the greedy loop.
    """
    dist = all_pairs_distances(neighbors)
    all_nodes = list(neighbors.keys())

    if n > len(all_nodes):
        raise ValueError(f"Requested cluster size {n} exceeds graph size {len(all_nodes)}")
    if n == 1:
        return [all_nodes[0]], 0

    # Seed: pick the pair of nodes with the smallest distance (==1 for
    # adjacent nodes), breaking ties by smallest total distance to all others.
    best_pair = None
    best_pair_score = inf
    for u in all_nodes:
        for v in neighbors[u]:
            if u >= v:
                continue
            score = sum(dist[u].get(w, inf) + dist[v].get(w, inf) for w in all_nodes)
            if score < best_pair_score:
                best_pair_score = score
                best_pair = [u, v]

    cluster = list(best_pair)
    in_cluster = set(cluster)

    # Running cost from each candidate to the current cluster
    candidate_cost = {}
    for node in all_nodes:
        if node not in in_cluster:
            candidate_cost[node] = sum(
                dist[node].get(c, inf) for c in cluster
            )

    # Greedily expand
    while len(cluster) < n:
        best_node = min(candidate_cost, key=candidate_cost.get)
        cluster.append(best_node)
        in_cluster.add(best_node)

        candidate_cost.pop(best_node)
        # Update remaining candidates with distance to the newly added node
        for node in list(candidate_cost):
            candidate_cost[node] += dist[node].get(best_node, inf)



    return order_cluster(cluster, dist), dist


# ── Convenience wrapper ───────────────────────────────────────────

def find_closest_cluster(
    n: int,
    edges: list[list[int]],
    neighbors: dict[int, set[int]],
    method: str = "auto",
) -> tuple[list[int], float]:
    """
    Find a cluster of `n` nodes minimising total pairwise distance.

    Parameters
    ----------
    n          : desired cluster size
    edges      : list of [u, v] edges (used only for validation here)
    neighbors  : adjacency dict  {node: set(neighbor_nodes)}
    method     : "exact", "greedy", or "auto"
                 ("auto" picks exact when C(|V|,n) <= 50 000)

    Returns
    -------
    (cluster_nodes, total_pairwise_distance)
    """
    num_nodes = len(neighbors)

    if method == "auto":
        # Rough estimate of combination count
        from math import comb
        method = "exact" if comb(num_nodes, n) <= 50000 else "greedy"

    if method == "exact":
        return find_closest_cluster_exact(n, neighbors)
    else:
        return find_closest_cluster_greedy(n, neighbors)
