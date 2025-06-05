Functions:
    build_interaction_graph(Q):
        Build adjacency list (interaction graph) from an upper-triangular QUBO matrix.
    find_cluster(graph, diff_nodes, start_node):
        Find a connected cluster of differing nodes using DFS/BFS starting from a seed node.
    isoenergetic_cluster_move(stateA, stateB, Q):
        Perform an ICM operation on two replicas (stateA, stateB) by identifying a cluster of 
        differing bits and swapping them between the two states.

Example:
    # Define a QUBO matrix (4x4 symmetric with non-zero couplings)
    Q = [
        [0,  1,  0,  0],
        [1,  0,  1,  0],
        [0,  1,  0,  1],
        [0,  0,  1,  0]
    ]
    # Two example states (replicas), represented as lists of bits (0/1):
    stateA = [0, 0, 0, 0]
    stateB = [0, 1, 1, 0]
    # Perform an isoenergetic cluster move on the two states:
    new_stateA, new_stateB = isoenergetic_cluster_move(stateA, stateB, Q)
    print(new_stateA, new_stateB)
"""
import random

def build_interaction_graph(Q):
    """
    Construct an interaction graph (adjacency list) from an upper-triangular QUBO matrix.
    
    The graph's nodes correspond to variables (indices of the Q matrix). An undirected edge 
    is created between any two variables i and j if the coupling Q[i][j] (or Q[j][i]) is non-zero.
    Only the upper triangular part of Q (i < j) is considered to avoid duplicate edges.
    
    Args:
        Q (List[List[float]] or numpy.ndarray): Symmetric QUBO/Ising matrix. Q[i][j] represents 
            the coupling between variable i and j (diagonal entries represent local fields).
            
    Returns:
        List[List[int]]: Adjacency list where index i contains a list of neighbor indices j 
        such that Q[i][j] (coupling between i and j) is non-zero.
    """
    n = len(Q)  # number of variables (dimension of Q)
    # Initialize an empty adjacency list (list of neighbor lists for each node)
    graph = [[] for _ in range(n)]
    # Iterate over upper-triangular part of Q (excluding diagonal) to find non-zero couplings
    for i in range(n):
        for j in range(i+1, n):
            # Check if coupling (i,j) is non-zero
            if Q[i][j] != 0:
                # Add edge to adjacency list (undirected graph)
                graph[i].append(j)
                graph[j].append(i)
    return graph

def find_cluster(graph, diff_nodes, start_node):
    """
    Find a connected cluster of differing nodes in the interaction graph, starting from a seed node.
    
    Uses a depth-first search (DFS) or breadth-first search (BFS) to collect all nodes that:
    - are in the set of differing nodes (diff_nodes), and 
    - are connected to the start_node through other differing nodes.
    
    This yields the connected component (cluster) of the graph that contains start_node, restricted 
    to the subgraph induced by diff_nodes.
    
    Args:
        graph (List[List[int]]): Adjacency list representation of the interaction graph (as returned by build_interaction_graph).
        diff_nodes (List[int] or Set[int]): Collection of node indices where two states differ.
        start_node (int): The starting node index (should be an element of diff_nodes) to grow the cluster from.
    
    Returns:
        List[int]: A list of node indices representing the connected cluster of differing nodes 
        that includes start_node.
    """
    # Ensure diff_nodes is a set for O(1) membership tests
    diff_set = set(diff_nodes)
    # Cluster will store all nodes in the connected component
    cluster = []
    # Use a stack for DFS (could also use a queue for BFS; result is the same cluster)
    stack = [start_node]
    # A set to keep track of visited nodes in the cluster search
    visited = {start_node}
    # Perform DFS/BFS
    while stack:
        node = stack.pop()
        cluster.append(node)
        # Explore all neighbors of the current node
        for neighbor in graph[node]:
            # Only consider neighbors that are in diff_nodes and not yet visited
            if neighbor in diff_set and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return cluster

def isoenergetic_cluster_move(stateA, stateB, Q):
    """
    Perform an Isoenergetic Cluster Move (ICM) on two replica states.
    
    The ICM algorithm finds a cluster of bits (spins) that differ between two states and swaps them,
    creating two new states. This move is "isoenergetic" in the sense that if stateA and stateB have 
    the same energy, swapping a cluster of differing bits will preserve that energy for both (hence 
    both states remain at the same energy after the swap).
    
    The procedure is:
      1. Identify all indices where stateA and stateB differ.
      2. Build the interaction graph of the problem using the Q matrix (non-zero couplings define edges).
      3. Restrict this graph to the subgraph of differing nodes. Pick a random starting node from the differing set.
      4. Find all connected nodes reachable from the start_node (within the subgraph of differing nodes) 
         using the interaction graph. This set of nodes forms the cluster.
      5. Swap the values of the bits/spins in this cluster between stateA and stateB.
    
    Args:
        stateA (List[int]/List[float]): State of the first replica (vector of 0/1 or -1/1 values).
        stateB (List[int]/List[float]): State of the second replica (same size as stateA).
        Q (List[List[float]] or numpy.ndarray): Symmetric QUBO/Ising matrix defining the problem couplings.
    
    Returns:
        Tuple[List, List]: A tuple (new_stateA, new_stateB) after performing the cluster swap. 
        The states are also modified in place.
    """
    # Identify indices where the two states differ
    diff_nodes = [i for i in range(len(stateA)) if stateA[i] != stateB[i]]
    # If there are no differences, no cluster move can be performed
    if not diff_nodes:
        # Nothing to swap, return the states unchanged
        return stateA, stateB
    # Build the adjacency list for the interaction graph from the Q matrix
    graph = build_interaction_graph(Q)
    # Choose a random starting node from the differing nodes
    start_node = random.choice(diff_nodes)
    # Find the full cluster of connected differing nodes starting from this node
    cluster = find_cluster(graph, diff_nodes, start_node)
    # Swap the bits/spins in the cluster between stateA and stateB
    for i in cluster:
        stateA[i], stateB[i] = stateB[i], stateA[i]
    # Return the new states (they are also modified in-place)
    return stateA, stateB

# Example usage demonstration
if __name__ == "__main__":
    # Example QUBO problem with 4 variables (fully symmetric matrix including diagonal local fields if any).
    # Here we define couplings between some pairs of variables (non-zero entries off-diagonal).
    Q = [
        [0,  1,  0,  0],  # Coupling between variable 0 and 1 is 1
        [1,  0,  1,  0],  # Coupling between variable 1 and 2 is 1 (and 0-1 is mirrored)
        [0,  1,  0,  1],  # Coupling between variable 2 and 3 is 1
        [0,  0,  1,  0]   # Coupling 2-3 is mirrored
    ]
    # Two example states (replicas) as lists. Using 0/1 to represent binary states for QUBO.
    # These states have differences at indices 1 and 2 (stateA has 0, stateB has 1 at those positions).
    stateA = [0, 0, 0, 0]
    stateB = [0, 1, 1, 0]
    print("Initial stateA:", stateA)
    print("Initial stateB:", stateB)
    # Perform an isoenergetic cluster move on the two states
    new_stateA, new_stateB = isoenergetic_cluster_move(stateA, stateB, Q)
    print("After ICM move, stateA:", new_stateA)
    print("After ICM move, stateB:", new_stateB)