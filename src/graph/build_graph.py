"""
    
    This file should be responsible for graph creation in the initial state, 


"""


from typing import Optional, Dict, List, Tuple




class GraphBuilder:
    """
    Constructs a graph from Node and Edge objects before handing off
    to a read-only Graph instance.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._root: Optional[str] = None

    def set_root(self, node: Node) -> None:
        """Set the root node."""
        self._root = node.id
        self._nodes[node.id] = node

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node

    def attach_edge(self, edge: Edge) -> None:
        """Attach an edge to the graph."""
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise ValueError("Both source and target nodes must be added before attaching an edge.")
        self._edges.append(edge)

    def build(self) -> Graph:
        """Return an immutable Graph object."""
        return Graph(nodes=self._nodes, edges=self._edges, root=self._root)
        return Graph(self._nodes, self._edges, self._root)