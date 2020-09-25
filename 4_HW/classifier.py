import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import stop_words
from nltk.tokenize import TreebankWordTokenizer
import random
import numpy as np
import pandas as pd
import pprint
import tokenize
from statistics import mean

#!from google.colab import drive
#!drive.mount('/content/gdrive')

DEBUG = 0
DATA_DEBUG = 1
BADLINE = 171891

# dictionaries to hold data read from files
movie_title = {}        # movie titles by movieId
movie_year = {}         # movie year by movieId
movie_genres = {}       # list of genre keywords, by movieId
movie_plot = {}         # movie plots by movieId
movie_imdb_rating = {}  # movie IMDb rating by movieId
user_ratings = {}       # list of (movieId, rating, timestamp) by movieId
movieIDs = []
movies_In_Dataset = set()
movie_with_features = {}
movie_bag_of_words = {}

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
                movieIDs.append(movieId)
                if userId in user_ratings:
                    user_ratings[userId].append(user_rating)
                else:
                    user_ratings[userId] = [user_rating]
            line_num += 1

from sklearn.metrics.pairwise import pairwise_distances


def build_dataset(numOfMovies):
    userID = 1
    dataset = []
    while userID <= numOfMovies:
        firstMovFlag=0
        secMovFlag=0
        # the random number should be just the len of items in the list
        #movID_1 = movieIDs[random.randrange(0, len(movieIDs), 1)]
        #movID_2 = movieIDs[random.randrange(0, len(movieIDs), 1)]
        firstMovRating = 0.0
        secMovRating = 0.0
        userRating = user_ratings[userID]
        movID_1 = random.randrange(0, len(userRating), 1)
        movID_2 = random.randrange(0, len(userRating), 1)
        """ for eachMovieRating in userRating:
            if eachMovieRating[0] == movID_1:
                firstMovRating = float(eachMovieRating[1])
                firstMovFlag += 1
            if eachMovieRating[0] == movID_2:
                secMovRating = float(eachMovieRating[1])
                secMovFlag += 1
            if firstMovFlag and secMovFlag:
                break """
        for i, eachMovieRating in enumerate(userRating):
            if i == movID_1:
                firstMovRating = float(eachMovieRating[1])
                firstMovFlag += 1
                movies_In_Dataset.add(eachMovieRating[0])
            elif i == movID_2:
                secMovRating = float(eachMovieRating[1])
                secMovFlag += 1
                movies_In_Dataset.add(eachMovieRating[0])
            if firstMovFlag and secMovFlag:
                break
        rating_diff = float(abs(float(firstMovRating) - float(secMovRating)))
        dataset.append([userID, movID_1, movID_2, rating_diff])
        #movies_In_Dataset.update([movID_1, movID_2])
        userID += 1
    return dataset

# Compute word vectors for movie genres, titles, and plots, for movies in dataset.
def extract_text_features(dataset_movieIDs):
    word_list_bag = set()

    for each_movie_id in dataset_movieIDs:
        #word_list_bag.append(movie_plot[each_movie_id].split())
        #tempList = set()
        tempList = []
        movie_words = []
        for eachWord in movie_plot[each_movie_id].split():
            #word_list_bag.append(eachWord)
            word_list_bag.add(eachWord)
            #tempList.add(eachWord)
            tempList.append(eachWord)
            """ tempList.add(movie_title[each_movie_id])
            for eachGenre in movie_genres[each_movie_id]:
                tempList.add(eachGenre)
            tempList.add(movie_year[each_movie_id]) """
        count_vect = CountVectorizer()
        tokenizer = TreebankWordTokenizer()
        count_vect.set_params(tokenizer=tokenizer.tokenize)
        count_vect.set_params(stop_words='english')
        #print(stop_words.ENGLISH_STOP_WORDS)
        count_vect.set_params(ngram_range=(1,2))
        count_vect.set_params(max_df=0.5)
        count_vect.set_params(min_df=1)
        movie_words = count_vect.fit_transform(tempList)
        if each_movie_id not in movie_with_features:
            movie_with_features[each_movie_id] = [movie_words, movie_title[each_movie_id], movie_genres[each_movie_id], movie_year[each_movie_id]]
            #movie_with_features[each_movie_id] = [movie_words, count_vect.fit_transform(movie_title[each_movie_id]), count_vect.fit_transform(movie_genres[each_movie_id]), count_vect.fit_transform(movie_year[each_movie_id])]
            movie_bag_of_words[each_movie_id] = movie_words
    #print(word_list_bag)


def build_training_set(in_data):
    # normalize counts based on document length
    # weight common words less (is, a, an, the)
    #tfidf_transformer = TfidfTransformer()
    #X_tfidf = tfidf_transformer.fit_transform(np.reshape(in_data,(-1,2)))
    #X_tfidf = tfidf_transformer.fit_transform(np.array([in_data]).reshape((-1, 2)))
    #X_tfidf = tfidf_transformer.fit_transform([in_data])
    #clf = MultinomialNB().fit(X_tfidf, movie_imdb_rating)
    #scores = cross_val_score(clf, X_tfidf, movie_imdb_rating, cv=3)
    pass

def naive_rank_train(in_data):
    new_Rank_Movie_List = {}
    userID = 1
    #while userID <= len(in_data):
    testing = 30
    while userID <= len(in_data):
        for i, eachMovieRating in enumerate(in_data[userID]):
            temp_movieID = eachMovieRating[0]
            temp_movie_rating = eachMovieRating[1]
            if temp_movieID in new_Rank_Movie_List:
                new_Rank_Movie_List[temp_movieID].append(temp_movie_rating)
            else:
                new_Rank_Movie_List[temp_movieID] = [temp_movie_rating]
        userID += 1
    #TODO Take averages and then put the movieId, rating in a tuple
    return new_Rank_Movie_List

def compare_ratings(userRated, imbdRated):
    return imbdRated - userRated

def naive_rank_test(dataset):
    #TODO do the conversion to tuple here?
    new_movie_list = []
    for eachID in dataset:
        new_movie_list.append((eachID, mean(dataset[eachID])))
    new_movie_list.sort(key = lambda x: x[1], reverse=True)
    #now print list based on sorted tuple
    
    print("::::::::::::::::::::::::NEW RANKINGS::::::::::::::::::::::::")
    for i,eachTup in enumerate(new_movie_list):
        if i >= 200:
            break
        print('RANKED #' + str(i+1) + " " + movie_title[eachTup[0]] + "| Score of " + str(eachTup[1]) + " | IMDB Difference: " + str(compare_ratings(eachTup[1], movie_imdb_rating[eachTup[0]])))


def main():
    global movie_title, ranking_limit
    print("Reading data...", flush=True)
    read_data()
    #print("titles", movie_title)
    #print("years", movie_year)
    #print("genres", movie_genres)
    print("plots", movie_plot)
    print("\nratings", movie_imdb_rating)
    print("\nuser_ratings", user_ratings)
    print("\nuser_ratings", user_ratings[1])
    print("\nuser_ratings", user_ratings[2])
    #print("\nuser_ratings", user_ratings[10])
    #print(type(user_ratings[1][0]))
    dataset = build_dataset(200)

    text_features = extract_text_features(movies_In_Dataset)
    
    if DATA_DEBUG:
        print("dataset:\n")
        print(dataset)

        print("Movies in Dataset")
        print(list(movies_In_Dataset))
    
    #build_training_set(movie_with_features)
    #build_training_set(movie_bag_of_words)
    #build_training_set(dataset)  # Construct sparse matrix X of feature vectors for each example in dataset. Construct target classes y a    nd sample weights w for each example in dataset.
    
    # Sample for testing
    #example = dataset[0] 
    #fv = generate_feature_vector(example) # feature vector generation(bag of words for our features
    
    # Naive Ranking
    #classifier = MultinomialNB()
    #classifier = naive_rank_train() # This is where we train our model based on the rating of each user
    #naive_ranking = naive_rank_test(dataset, classifier[:ranking_limit]  # Return list of (movieId, score) pairs sorted in decreasing ord    er by score. The classifier is used to predict preference between each pair of movies.
    #imdb_ranking = get_imdb_ranking(naive_ranking)
    #dist = compare_rankings(naive_ranking,imdb_ranking) # compare the ratings that we generated based on users with imdb ranking
    movie_rank_data = naive_rank_train(user_ratings)

    naive_rank_test(movie_rank_data)

main()
