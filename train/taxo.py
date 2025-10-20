'''taken from the official TEMP paper implementation'''

import networkx as nx

class TaxStruct(nx.DiGraph):
    def __init__(self, edges):
        # edges should be (child, parent) for correct graph direction
        super().__init__(edges)
        self.check_useless_edge()
        self._root = ""
        # Find root: node with in-degree 0
        roots = [node for node in self.nodes if self.in_degree(node) == 0]
        if not roots:
            raise ValueError("Taxonomy graph has no root (no node with in-degree 0).")
        if len(roots) > 1:
             print(f"Warning: Taxonomy graph has multiple roots: {roots}. Using the first one found: {roots[0]}")
        self._root = roots[0]


        self._node2path = dict()
        # Calculate shortest path from root to all nodes
        try:
            shortest_paths = nx.shortest_path(self, source=self._root)
            for node in self.nodes.keys():
                 if node in shortest_paths:
                    self._node2path[node] = list(reversed(shortest_paths[node]))
                 else:
                     print(f"Warning: Node '{node}' is not reachable from the root '{self._root}'. It will not have a path stored.")
                     # Handle nodes not reachable from root if necessary
                     self._node2path[node] = [] # Assign empty path or handle differently
        except nx.NetworkXNoPath as e:
            print(f"Error computing shortest paths from root: {e}")
            # This exception is less likely now with the individual node check, but kept for safety.
            pass # Handle the error as needed


        self.leaf_nodes = self.all_leaf_nodes()

    def check_useless_edge(self):
        """
        delete useless edges (edges where child has multiple parents and there's a path between those parents)
        """
        bad_edges = []
        for node in self.nodes:
            if self.in_degree(node) <= 1:
                continue
            parents = list(self.predecessors(node))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1 = parents[i]
                    p2 = parents[j]
                    # Check if there's a path from one parent to the other
                    if nx.has_path(self, p1, p2):
                        bad_edges.append((p2, node)) # Assuming p1 is a more direct parent
                    elif nx.has_path(self, p2, p1):
                        bad_edges.append((p1, node)) # Assuming p2 is a more direct parent

        # Remove duplicate bad edges before removing from graph
        bad_edges = list(set(bad_edges))
        self.remove_edges_from(bad_edges)
        if bad_edges:
            print(f"Removed {len(bad_edges)} useless edges.")


    def all_leaf_nodes(self):
        # Leaf nodes are nodes with out-degree 0
        return [node for node in self.nodes.keys() if self.out_degree(node) == 0]

    def get_parents(self, node):
        """Get direct parents of a node"""
        return list(self.predecessors(node))

    @property
    def node2path(self):
        return self._node2path

    @property
    def root(self):
        return self._root
