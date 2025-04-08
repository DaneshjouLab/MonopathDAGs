# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0


"""
This file should contian an edge protocol. 

"""

from typing import Union,Protocol
from ..node.node_protocol import NodeProtocol


class EdgeProtocol(Protocol):
    "Edge Protocol"
    def __init__(self, source:Union[NodeProtocol], destination:Union[NodeProtocol]):
        self.source = source
        self.target = destination
    def get_data(self) -> Union[dict, None]:
        pass
    def __repr__(self) -> str:
        return f"Edge({self.source}, {self.target})"



class OrderedEdge(EdgeProtocol):
    "Ordered Edge"
    def __init__(self, source:Union[NodeProtocol],
                 destination:Union[NodeProtocol],
                 order_index:Union[int]):
        super().__init__(source, destination)
        self.order_index = order_index
    def get_data(self) -> Union[dict, None]:
        pass
    def __repr__(self) -> str:
        return f"OrderedEdge({self.source}, {self.target}, {self.order_index})"

