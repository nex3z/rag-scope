import hashlib
from typing import List, Optional, TypedDict, Literal

import matplotlib as mpl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.documents import Document
from loguru import logger
from plotly.graph_objs import Figure

from rag_scope.retriever.base import Query


class Record(TypedDict):
    id: Optional[str]
    group: Optional[int]
    is_query: bool
    content_type: str
    content: str
    embedding: List[float]


def build_document_data_frame(
    documents: List[Document],
    doc_type: Literal['stored', 'retrieved'],
    group: Optional[int] = None,
) -> pd.DataFrame:
    records = []
    for document in documents:
        metadata = document.metadata
        record: Record = {
            'id': metadata['id'] if 'id' in metadata else build_id_from_content(document.page_content),
            'group': group,
            'is_query': False,
            'content_type': doc_type,
            'content': document.page_content,
            'embedding': metadata['embedding'],
        }
        records.append(record)
    df = pd.DataFrame(records)
    return df


def build_id_from_content(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def build_query_data_frame(
    query: Query,
    group: Optional[int] = None,
) -> pd.DataFrame:
    record: Record = {
        'id': None,
        'group': group,
        'is_query': True,
        'content_type': query.type,
        'content': query.query_content,
        'embedding': query.metadata['embedding'],
    }
    df = pd.DataFrame([record])
    return df


def plot(df_points: pd.DataFrame):
    df_points = __preprocess(df_points)
    if 'z' in df_points.columns:
        return plot_3d(df_points)
    else:
        return plot_2d(df_points)


def plot_2d(df_points: pd.DataFrame):
    fig = go.Figure()
    cmap = px.colors.qualitative.Plotly

    query_content_types = df_points.loc[df_points['is_query'] == True]['content_type'].unique()
    logger.info(f"query_content_types = {query_content_types}")

    for idx, query_content_type in enumerate(query_content_types):
        color = cmap[(idx + 1) % len(cmap)]

        df_query = df_points.loc[(df_points['is_query'] == True) & (df_points['content_type'] == query_content_type)]
        fig.add_trace(
            go.Scatter(
                x=df_query['x'],
                y=df_query['y'],
                mode='markers',
                name=query_content_type,
                text=df_query['content'],
                marker=dict(symbol='cross', color=color),
            )
        )

        groups = df_query['group'].dropna().unique()
        logger.info(f"groups = {groups}")

        df_retrieved = df_points.loc[(df_points['is_query'] == False) & (df_points['group'].isin(groups))]
        print(f" len(df_retrieved)   ={ len(df_retrieved) }")
        if len(df_retrieved) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_retrieved['x'],
                    y=df_retrieved['y'],
                    mode='markers',
                    name='retrieved',
                    text=df_retrieved['content'],
                    marker=dict(color=color),
                )
            )

    df_stored = df_points.loc[df_points['is_query'] == False]
    fig.add_trace(
        go.Scatter(
            x=df_stored['x'],
            y=df_stored['y'],
            mode='markers',
            name='stored',
            text=df_stored['content'],
            marker=dict(symbol='circle-open', color=cmap[0]),
        )
    )

    __update_legend(fig)
    return fig


def plot_3d(df_points: pd.DataFrame):
    fig = go.Figure()
    cmap = px.colors.qualitative.Plotly

    query_content_types = df_points.loc[df_points['is_query'] == True]['content_type'].unique()
    logger.info(f"query_content_types = {query_content_types}")

    for idx, query_content_type in enumerate(query_content_types):
        color = cmap[(idx + 1) % len(cmap)]

        df_query = df_points.loc[(df_points['is_query'] == True) & (df_points['content_type'] == query_content_type)]
        fig.add_trace(
            go.Scatter3d(
                x=df_query['x'],
                y=df_query['y'],
                z=df_query['z'],
                mode='markers',
                name=query_content_type,
                text=df_query['content'],
                marker=dict(symbol='cross', color=color),
            )
        )

        groups = df_query['group'].dropna().unique()
        logger.info(f"groups = {groups}")

        df_retrieved = df_points.loc[(df_points['is_query'] == False) & (df_points['group'].isin(groups))]
        print(f" len(df_retrieved)   ={ len(df_retrieved)}")
        if len(df_retrieved) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=df_retrieved['x'],
                    y=df_retrieved['y'],
                    z=df_retrieved['z'],
                    mode='markers',
                    name='retrieved',
                    text=df_retrieved['content'],
                    marker=dict(color=color),
                )
            )

    df_stored = df_points.loc[df_points['is_query'] == False]
    fig.add_trace(
        go.Scatter3d(
            x=df_stored['x'],
            y=df_stored['y'],
            z=df_stored['z'],
            mode='markers',
            name='stored',
            text=df_stored['content'],
            marker=dict(symbol='circle-open', color=cmap[0]),
        )
    )

    __update_legend(fig)

    # fig = px.scatter_3d(
    #     df_points,
    #     x='x', y='y', z='z',
    #     hover_data=['content'],
    #     color='content_type',
    #     # color_discrete_map=DEFAULT_COLOR_DISCRETE_MAP,
    #     symbol='is_query',
    #     symbol_map={
    #         True: 'cross',
    #         False: 'circle-open',
    #     },
    # )
    #
    # ax_style = {
    #     'showbackground': False,
    #     'showgrid': True,
    #     'gridcolor': 'rgb(204, 204, 204)',
    #     'gridwidth': 1.5,
    #     'zeroline': True,
    # }
    # scene = {
    #     'xaxis': ax_style,
    #     'yaxis': ax_style,
    #     'zaxis': ax_style
    # }
    # fig.update_layout(scene=scene)
    # __update_legend(fig)
    return fig


def __update_legend(fig: Figure):
    fig.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        y=0,
        xanchor='right',
        x=1
    ))


def __preprocess(df_points: pd.DataFrame) -> pd.DataFrame:
    df_points = df_points.copy()

    df_points['content'] = df_points['content'].str.wrap(50)
    df_points['content'] = df_points['content'].apply(lambda x: x.replace('\n', '<br>'))

    return df_points


def main():
    cmap = mpl.colormaps['tab10']
    print(cmap(20))
    px.colors.qualitative.Alphabet()


if __name__ == '__main__':
    main()
