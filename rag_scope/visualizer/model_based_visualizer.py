from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from loguru import logger
from numpy.typing import NDArray

from rag_scope.retriever.base import Retrieval
from rag_scope.visualizer.base_retrieval_visualizer import BaseRetrievalVisualizer
from rag_scope.visualizer.plot_utils import plot, build_document_data_frame, build_query_data_frame


class ModelBasedVisualizer(BaseRetrievalVisualizer, ABC):
    def __init__(self, stored: List[Document], random_state: Optional[int] = None):
        self.df_stored = build_document_data_frame(stored, doc_type='stored')
        self.stored_embeddings = np.array(self.df_stored['embedding'].to_list())
        self.random_state = random_state

    @abstractmethod
    def transform(self, embeddings: NDArray) -> NDArray:
        pass

    def render(
        self,
        retrievals: List[Retrieval],
        limit: Optional[int] = None,
    ):
        df_stored = self.df_stored
        if (limit is not None) and (limit < len(df_stored)):
            df_stored = df_stored.sample(limit, random_state=self.random_state)

        df_queries, df_retrievals = [], []
        for group, retrieval in enumerate(retrievals):
            df_queries.append(build_query_data_frame(retrieval.query, group=group))
            df_retrievals.append(build_document_data_frame(retrieval.documents, doc_type='retrieved', group=group))

        df_data = pd.concat([df_stored, *df_queries])
        embeddings = np.array(df_data['embedding'].to_list())
        logger.info(f"embeddings.shape = {embeddings.shape}")

        points = self.transform(embeddings)
        logger.info(f"points.shape = {points.shape}")

        _, n_dims = points.shape
        if n_dims == 2:
            point_fields = ['x', 'y']
        elif n_dims == 3:
            point_fields = ['x', 'y', 'z']
        else:
            raise ValueError(f"Invalid number of dimensions: {n_dims}")

        for idx, field in enumerate(point_fields):
            df_data[field] = points[:, idx]

        # logger.info(f"df_data = \n{df_data}")

        df_retrievals_with_points = []
        for df_retrieval in df_retrievals:
            if len(df_retrieval) == 0:
                continue
            df_retrieval = df_retrieval.merge(df_data[['id', *point_fields]], on='id', how='left')
            df_retrievals_with_points.append(df_retrieval)

        df_data = pd.concat([df_data, *df_retrievals_with_points])

        return plot(df_data)
