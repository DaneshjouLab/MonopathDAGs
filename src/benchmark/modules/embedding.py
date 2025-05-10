# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""This module contains the TrajectoryEmbedder class for embedding patient trajectories."""

# Standard library imports
from typing import Optional

# Third-party library imports
import torch
from transformers import AutoTokenizer, AutoModel
import networkx as nx

# Local application imports
from .logging_utils import setup_logger
from .config import TRAJECTORY_EMBEDDING_MODEL

logger = setup_logger(__name__)


class TrajectoryEmbedder:
    """
    Embeds a patient trajectory graph by pooling embeddings of textual node content.
    Uses a transformer model (e.g., Bio_ClinicalBERT) and extracts node-level text
    based on a configurable path (e.g., ["data", "commentary"]).

    Attributes:
        model_name (str): Hugging Face model ID used for embedding.
        text_path (list[str]): Path to the text field in node
            attributes (e.g., ["data", "commentary"]).
        device (str): Computation device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_name=TRAJECTORY_EMBEDDING_MODEL,
        text_path=None,
        device=None,
    ):
        """
        Initializes the TrajectoryEmbedder with a specified model and text path.
        Args:
            model_name (str): Hugging Face model ID used for embedding.
            text_path (list[str]): Path to the text field in node
                attributes (e.g., ["data", "commentary"]).
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.text_path = text_path or ["content"]  # Default to flat structure
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_path = text_path
        logger.info("Loading embedding model: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> torch.Tensor:
        """"Embeds a single text string using the transformer model.
        Args:
            text (str): Text to be embedded.
        Returns:
            torch.Tensor: The embedding of the text.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0][0]

    def extract_text(self, data: dict) -> Optional[str]:
        """Extracts text from the node attributes based on the specified path.
        Args:
            data (dict): Node attributes.
        Returns:
            Optional[str]: Extracted text or None if not found.
        """
        try:
            for key in self.text_path:
                data = data[key]
            return data if isinstance(data, str) else None
        except (KeyError, TypeError):
            return None

    def embed_graph(self, graph: nx.DiGraph) -> Optional[torch.Tensor]:
        """Embeds a graph by pooling the embeddings of its nodes.
        Args:
            graph (nx.DiGraph): The graph to be embedded.
        Returns:
            Optional[torch.Tensor]: The pooled embedding of the graph.
        """
        texts = []
        for _, data in graph.nodes(data=True):
            t = self.extract_text(data)
            if t:
                texts.append(t)

        if not texts:
            logger.warning("No valid node texts found for embedding.")
            return None

        node_embs = torch.stack([self.embed_text(t) for t in texts])
        return node_embs.mean(dim=0).cpu()
