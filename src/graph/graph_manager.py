# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0


"""
Graph manager for building and managing graphs."""



from .node.node_protocol import NodeProtocol as Node
from .edge.edge_protocol import EdgeProtocol as Edge
from .build_graph import GraphBuilder



class GraphManager:
    """
    TODO:
    [ ] add support for single graph instance
    [ ] create add node 
    [ ] track pointer. 
    [ ] create_branch
    [ ] 
    """
    def __init__(self, startPoint: Node, graph_builder:GraphBuilder):
        """_summary_

        Args:
            startPoint (Node): this can be any point you start, root or current pointer
            graph_builder (GraphBuilder): the thing, 
The follwoing should be done tin tkhfm e


        """
        

        pass
    def verf
    

class DynamicDataGraphManager(GraphManager):
    """_summary_

    Args:
        GraphManager (_type_): _description_
    """
    def __init__(self, root):
        super().__init__(root)