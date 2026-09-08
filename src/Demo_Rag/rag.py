from utils import (
    retrieve,
    pprint,
    read_dataframe
)


NEWS_DATA = read_dataframe("news_data_dedup.csv")


def query_news(indices):
    
    output = [NEWS_DATA[index] for index in indices]

    return output


def get_relevant_data(query: str, top_k: int = 5) -> list[dict]:
    
    relevant_indices = retrieve(query=query, top_k=top_k)
    relevant_data = query_news(relevant_indices)

    return relevant_data