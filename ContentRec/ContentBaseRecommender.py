# This program is showing content based recommender system based on movie description
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBaseRecomender:
    """
    this is content base recommender class
    recommendation based on movie similarity, check movie description and vectorized it and find similarity.
    file link https://www.kaggle.com/rounakbanik/the-movies-dataset/downloads/movies_metadata.csv/7
    """

    def __init__(self, data, dataframe=None):
        """
        initiate data frame
        :param data:
        :param dataframe:
        """
        if dataframe:
            self.df_movie = dataframe
        else:
            self.df_movie = pd.read_csv(data)

    def modify_dataframe(self):
        """
        keep only required data
        """
        # Only keep those features that we require
        self.df_movie = self.df_movie[
            ['title', 'genres', 'release_date', 'runtime', 'vote_average', 'vote_count', 'id', 'overview']]
        # genre operations
        self.df_movie['genres'] = self.df_movie['genres'].fillna('[]')
        self.df_movie['genres'] = self.df_movie['genres'].apply(literal_eval)
        # convert genre dictionary to list
        self.df_movie['genres'] = self.df_movie['genres'].apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

        # also remove NaN from "overview" field
        self.df_movie['overview'] = self.df_movie['overview'].fillna('')

    def create_tf_idf_matrix(self):
        """
        creating tf-idf matrix based on movie "overview", note here we are removing stop words
        stop words are common english words like "the","a","in","on" etc.
        """
        # clean dataframe
        self.modify_dataframe()

        tfidf = TfidfVectorizer(stop_words='english')

        # Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
        # This matrix is contains all word vectors.one dimension of this matrix should be the same as df_movie data frame
        tfidf_matrix = tfidf.fit_transform(self.df_movie['overview'])

        # Compute the cosine similarity matrix THIS STEP IS HEAVILY COMPUTE BASED. it required atleast 32 GB RAM.
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        return self.cosine_sim

    def create_index_mapping(self):
        """
        this function is creating mapping between title and index for east title search
        """
        print(self.df_movie)
        self.indices = pd.Series(self.df_movie.index, index=self.df_movie['title']).drop_duplicates()
        return self.indices

    def get_recommendation(self, movie_title, top=10):
        """
        get recommendation based on movie title, enter movie title and get top similar movie based on description
        """
        # Obtain the index of the movie that matches the title
        cosine_sim = self.create_tf_idf_matrix()
        # create index and cosine matrix
        indices = self.create_index_mapping()

        try:
            movie_index = indices[movie_title]

            # Get the pairwsie similarity scores of all movies with that movie from cosine index
            sim_scores = list(enumerate(cosine_sim[movie_index]))

            # Sort the movies based on the cosine similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the top most similar movies. Ignore the first movie.
            sim_scores = sim_scores[1:top]

            # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]

            # return top similar movies
            return self.df_movie['title'].iloc[movie_indices]

        except IndexError:
            print("Movie that entered for recommendation is not found or may be mispelled?")
