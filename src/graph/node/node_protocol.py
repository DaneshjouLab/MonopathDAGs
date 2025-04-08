# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0


"""
This file should contian an edge protocol. 

# TODO: implement the following, 

"""
from typing  import Protocol,Union
from dataclasses import dataclass
from uuid import UUID

@dataclass
class NodeData:
    """ Class for indicting data stored in node. """
    # TODO must conform to a structure that is recursivley a dictionary- done in base data move to appropriate folder, 






class NodeProtocol(Protocol):
    """


    Args:
        Protocol (_type_): _description_
    """
    def __init__(self, id:Union[int,UUID]):

        self.id = id
        self.neighbors = set()

    def get_data(self)-> Union[dict, None, NodeData]:
        "get the data of the node"
    