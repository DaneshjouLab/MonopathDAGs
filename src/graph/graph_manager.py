# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0


"""
Graph manager for building and managing graphs."""

from dataclasses import dataclass, fields, is_dataclass, MISSING
from typing import Any, Type, get_origin, get_args
from typing_extensions import Self
from .node.node_protocol import NodeProtocol as Node
from .edge.edge_protocol import EdgeProtocol as Edge

@dataclass
class BaseData:
    """
    Base class for dataclasses that want to support construction
    from unstructured dicts via the from_attrs() method.

    Supports:
    - Nested dataclass construction
    - Lists of nested dataclasses
    - Skipping missing fields
    - Ignoring unknown fields

    Pylint-safe: handles missing fields, unused args, and unknown keys gracefully.
    """

    @classmethod
    def from_attrs(cls: Type[Self], attrs: dict[str, Any]) -> Self:
        """
        Create an instance of this dataclass from a dictionary of attributes.

        Args:
            attrs (dict): Input dictionary of attributes.

        Returns:
            Instance of cls constructed from the filtered and typed attributes.

        Notes:
            - Nested dataclass fields are recursively constructed.
            - List fields containing dataclasses are supported.
            - Extra fields not defined in the dataclass are ignored.
        """

        # Dictionary to collect processed, type-safe constructor arguments
        instance_data = {}

        # Iterate over all declared fields of the dataclass
        for field in fields(cls):
            field_name = field.name
            field_type = field.type

            # Attempt to retrieve the value for this field
            value = attrs.get(field_name, MISSING)

            # Skip the field if not present in the input dictionary
            if value is MISSING:
                continue

            # Determine the type origin (e.g., list, Union) and type arguments
            origin = get_origin(field_type)
            args = get_args(field_type)

            # Case 1: Field is a nested dataclass and value is a dict
            if is_dataclass(field_type) and isinstance(value, dict):
                instance_data[field_name] = field_type.from_attrs(value)

            # Case 2: Field is a list of dataclasses (e.g., List[MyType])
            elif origin is list and args and is_dataclass(args[0]):
                nested_type = args[0]
                instance_data[field_name] = [
                    nested_type.from_attrs(v) if isinstance(v, dict) else v
                    for v in value
                ]

            # Case 3: Field is a primitive or already well-formed value
            else:
                instance_data[field_name] = value

        # Return an instance of the dataclass with validated inputs
        return cls(**instance_data)


class GraphManager:
    """
    TODO:
    [ ] add support for single graph instance
    [ ] create add node 
    [ ] track pointer. 
    [ ] create_branch
    [ ] 
    """
    def __init__(self, root: Node):
        
        pass


class DynamicDataGraphManager(GraphManager):
    """_summary_

    Args:
        GraphManager (_type_): _description_
    """
    def __init__(self, root):
        super().__init__(root)