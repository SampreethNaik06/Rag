from utils import (
    retrieve,
    pprint,
    read_dataframe
)


NEWS_DATA = read_dataframe("news_data_dedup.csv")


def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
    output = [NEWS_DATA[index] for index in indices]

    return output


def get_relevant_data(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve and return the top relevant data items based on a given query.

    This function performs the following steps:
    1. Retrieves the indices of the top 'k' relevant items from a dataset based on the provided `query`.
    2. Fetches the corresponding data for these indices from the dataset.

    Parameters:
    - query (str): The search query string used to find relevant items.
    - top_k (int, optional): The number of top items to retrieve. Default is 5.

    Returns:
    - list[dict]: A list of dictionaries containing the data associated
      with the top relevant items.
    """
    relevant_indices = retrieve(query=query, top_k=top_k)
    relevant_data = query_news(relevant_indices)

    return relevant_data