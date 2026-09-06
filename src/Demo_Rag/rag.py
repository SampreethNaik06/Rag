from utils import (
    retrieve, 
    pprint, 
    generate_with_single_input, 
    read_dataframe
)
import unittests


NEWS_DATA = read_dataframe("news_data_dedup.csv")
pprint(NEWS_DATA[9:11])

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


    
# Fetching some indices
indices = [3, 6, 9]
pprint(query_news(indices))


# Let's test the retrieve function!
indices = retrieve("Concerts in North America", top_k = 1)
print(indices)


# Now let's query the corresponding news_
retrieved_documents = query_news(indices)
pprint(retrieved_documents)

# GRADED CELL 

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
    ### START CODE HERE ###

    # Retrieve the indices of the top_k relevant items given the query
    relevant_indices = retrieve(query=query, top_k=top_k)  # @REPLACE EQUALS None

    # Obtain the data related to the items using the indices from the previous step
    relevant_data = query_news(relevant_indices)  # @REPLACE EQUALS None

    ### END CODE HERE
    
    return relevant_data


query = "Greatest storms in the US"
relevant_data = get_relevant_data(query, top_k = 1)
pprint(relevant_data)

# Run this cell to perform several tests on your function. If you receive "All test passed!" it means that your solution will likely pass the autograder too.
unittests.test_get_relevant_data(get_relevant_data)