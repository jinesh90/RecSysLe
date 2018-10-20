"""
gives simple top movies based on avg user rating and number of ratings.
"""
import pandas as pd

class SimpleRecommender:
    """
    give top movies based on no of votes and avg rating
    file link https://www.kaggle.com/rounakbanik/the-movies-dataset/downloads/movies_metadata.csv/7
    """
    def __init__(self, data=None, dataframe=None):
        """
        :param data: provide data path
        :param dataframe: or provide data frame
        """
        # Load movies into csv data frame
        if dataframe is not None:
            self.df_movie = dataframe
        else:
            self.df_movie = pd.read_csv(data)

    def weighted_formula(self, data_frame, m, C):
        """
        weighted rating
        WR = (v/v+m)xR + (m/v+m)xC
        where
        v = # of votes
        m = is the minimum number of votes required for the movie to be in the chart (the prerequisite)
        R = mean rating of movie
        C is mean rating of all movies.
        :param data_frame: movie data frame
        :return: formula
        """
        v = data_frame['vote_count']
        r = data_frame['vote_average']
        # Compute the weighted score
        return (v / (v + m) * r) + (m / (m + v) * C)

    def get_recommendation(self, top=250):
        """
        simple recommendation engine
        :arg top # of top recommendation.
        :return: data frame with
        """
        # higher the selected value means higher # votes.for the good result select higher percentile
        selected_value = self.df_movie['vote_count'].quantile(0.85)

        # now filter movies who has runtime between 45 mins to 300 mins to qualify movies.
        new_df_movies = self.df_movie[(self.df_movie['runtime'] >= 45) & (self.df_movie['runtime'] <= 300)]

        # consider only movies that has significant votes, selected value can be percentile or random number
        new_df_movies = new_df_movies[new_df_movies['vote_count'] > selected_value]

        # calculate mean
        vote_avg_mean = self.df_movie['vote_average'].mean()

        # generate new score based on formula

        new_df_movies['score'] = new_df_movies.apply(self.weighted_formula,
                                                     args=(selected_value, vote_avg_mean),
                                                     axis=1)

        return new_df_movies.sort_values(by=['score'], ascending=False)[0:top]


# run example, uncomment following code
# S = SimpleRecommender("/home/jinesh/Desktop/movies_metadata.csv")
# print(S.get_recommendation(top=25))
