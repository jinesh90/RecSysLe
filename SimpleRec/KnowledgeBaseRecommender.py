import pandas as pd
from ast import literal_eval
from SimpleRecommender import SimpleRecommender as SimpleReco


class KnowledgeBaseRecommender:
    """
    recommendation based on user given data, knowledge base recommendation.
    file link https://www.kaggle.com/rounakbanik/the-movies-dataset/downloads/movies_metadata.csv/7
    user can select
    i)  start year e.g 1990
    ii) end year e.g 2010
    iii)genre e.g Drama,Action, Horror etc
    """
    def __init__(self,data,dataframe=None):
        """
        initiate data frame
        :param data:
        :param dataframe:
        """
        if dataframe:
            self.df_movie = dataframe
        else:
            self.df_movie = pd.read_csv(data)

    @staticmethod
    def get_release_year(x):
        return int(x.split('-')[0])

    def modify_dataframe(self):
        """
        modify data frame, drop unnecessary columns
        :return:
        """
        # Only keep those features that we require
        self.df_movie = self.df_movie[['title', 'genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

        # fill release date in case of NaN
        self.df_movie['release_date'] = self.df_movie['release_date'].fillna('0-0-0')

        # set release year
        self.df_movie['release_year'] = self.df_movie['release_date'].apply(self.get_release_year)

        # drop release data as only release year required
        self.df_movie = self.df_movie.drop('release_date', axis=1)

        # genre operations
        self.df_movie['genres'] = self.df_movie['genres'].fillna('[]')
        self.df_movie['genres'] = self.df_movie['genres'].apply(literal_eval)

        # convert genre dictionary to list
        self.df_movie['genres'] = self.df_movie['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

        # select first genre among list
        self.df_movie['genres'] = self.df_movie['genres'].apply(lambda x: x[0] if x else [])

    def get_recommendation(self,start_year=0,end_year=0,genre="",top=50):
        """

        :param start_year: movie release year
        :param end_year: movie release year
        :param genre: ["Action","Drama","Comedy" etc]
        :param top : top 50 movies
        :return: top movie list
        """
        if start_year == 0 or end_year == 0 or genre == "":
            print("Please provide all necessary data")

        # before operation modify dataframe
        self.modify_dataframe()

        # copy newly generated dataframe
        movies = self.df_movie.copy()
        movies = movies[(movies['release_year'] >= start_year) & (movies['release_year'] <= end_year) & (movies['genres'] == genre)]

        # pass modified dataframe to simple reco for get top data based on avg star and # of comment
        S = SimpleReco(dataframe=movies)
        return S.get_recommendation(top=top)

# example usage
# K = KnowledgeBaseRecommender('/home/jinesh/Desktop/movies_metadata.csv')
# print(K.get_recommendation(2000,2018,"Horror", top=10))
