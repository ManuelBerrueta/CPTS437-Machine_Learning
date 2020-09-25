import csv
import nltk
nltk.download()
#!from google.colab import drive
#!drive.mount('/content/gdrive')

DEBUG = 0
BADLINE = 171891

# dictionaries to hold data read from files
movie_title = {}        # movie titles by movieId
movie_year = {}         # movie year by movieId
movie_genres = {}       # list of genre keywords, by movieId
movie_plot = {}         # movie plots by movieId
movie_imdb_rating = {}  # movie IMDb rating by movieId
user_ratings = {}       # list of (movieId, rating, timestamp) by movieId


def read_data():
    global movie_title, movie_year, movie_genres, movie_plot, movie_imdb_rating, user_ratings
    # read movie titles, years, and genres
    #!with open('/content/gdrive/My Drive/ML/HW/movies.csv') as csv_file:
    with open('movies.csv', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                movieId = int(row[0])
                title_year = row[1]
                genres = row[2]
                if DEBUG and movieId > BADLINE:
                    print(movieId, title_year, genres)
                temp_title_year = title_year[:-7]
                """ if temp_title_year[-1] == ' ':
                    temp_title_year = temp_title_year[0:-2]
                    movie_title[movieId] = temp_title_year[:-7]
                else:
                    movie_title[movieId] = title_year[:-7] """                
                movie_title[movieId] = title_year[:-7]
                if DEBUG and movieId > BADLINE:
                    print(movieId)
                    print(" title_year = " + title_year[-5:-1])
                movie_year[movieId] = int(title_year[-5:-1])
                if genres == "(no genres listed)":
                    movie_genres[movieId] = []
                else:
                    movie_genres[movieId] = genres.split('|')
            line_num += 1
    # read movie plots
    #!with open('/content/gdrive/My Drive/ML/HW/plots-imdb.csv') as csv_file:
    with open('plots-imdb.csv', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                movieId = int(row[0])
                plot = row[1]
                movie_plot[movieId] = plot
            line_num += 1
    # read movie IMDb ratings
    #!with open('/content/gdrive/My Drive/ML/HW/ratings-imdb.csv') as csv_file:
    with open('ratings-imdb.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                movieId = int(row[0])
                rating = float(row[1])
                movie_imdb_rating[movieId] = rating
            line_num += 1
    # read user ratings of movies
    #!with open('/content/gdrive/My Drive/ML/HW/ratings.csv') as csv_file:
    with open('ratings.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_num = 0
        for row in csv_reader:
            if line_num > 0:  # skip header
                userId = int(row[0])
                movieId = int(row[1])
                rating = float(row[2])
                timestamp = int(row[3])
                user_rating = (movieId, rating, timestamp)
                if userId in user_ratings:
                    user_ratings[userId].append(user_rating)
                else:
                    user_ratings[userId] = [user_rating]
            line_num += 1




def main():
    global movie_title, ranking_limit
    print("Reading data...", flush=True)
    read_data()
    #print("titles", movie_title)
    #print("years", movie_year)
    #print("genres", movie_genres)
    #print("plots", movie_plot)
    #print("\nratings", movie_imdb_rating)
    print("\nuser_ratings", user_ratings[3])


main()
