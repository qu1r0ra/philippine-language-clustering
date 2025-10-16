import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(reduced_df: pd.DataFrame) -> pd.DataFrame:
    sim_matrix = cosine_similarity(reduced_df)
    sim_df = pd.DataFrame(sim_matrix, index=reduced_df.index, columns=reduced_df.index)
    return sim_df
