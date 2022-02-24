#!/usr/bin/env python
# coding: utf-8

# #  ML MOVIE RECOMMENDER SYSTEM

# In[2]:


import numpy as np
import pandas as pd


# # DATA FETCHING

# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head()


# In[5]:


credits.head(1)['cast'].values    #to check values under one particular column for one particular entry


# # DATA PREPROCESSING

# In[6]:


movies = movies.merge(credits,on='title')  # merge the two datasets on the basis of "title" so that its easy to deal with


# In[7]:


movies.head()


# In[ ]:


# Filter the columns for model creation
# genres, id, keywords, title, overview, cast, crew


# In[8]:


movies['original_language'].value_counts()     #checking how many movies are in all the languages


# In[9]:


movies.info()             #to display the column names clearly 


# In[10]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]    #filtering and keeping only the relevant columns


# In[11]:


movies.head()


# In[15]:


movies.isnull().sum()       #to check which columns have null values


# In[14]:


movies.dropna(inplace=True )    #to drop the null values


# In[16]:


movies.duplicated().sum()      #to check duplicate entries


# In[17]:


movies.iloc[0].genres          #to display contents of row zero and column "genres"


# In[21]:


import ast    #to use ast.literal_eval for converting string into list


# In[24]:


def convert(obj):                  #this function will seprate the keywords of genres and store them into "L"
    L=[]
    for i in ast.literal_eval(obj):      #ast.literal_eval coverts the input(obj) which is a string into a list so that we can read it seprately
        L.append(i['name'])
    return L


# In[26]:


movies['genres'] = movies['genres'].apply(convert)  #calling the convert function to edit the genres column's format


# In[27]:


movies.head()   #check if the format has been changed


# In[29]:


movies['keywords'] = movies['keywords'].apply(convert)  #to change the format of "keywords" column as it is also in the same format as the genres


# In[30]:


movies.head()


# In[32]:


def convertcrew(obj):                  #this function will seprate the top 3 actors of from the cast column and  store them into "L"
    L=[]
    counter=0
    for i in ast.literal_eval(obj):   #ast.literal_eval coverts the input(obj) which is a string into a list so that we can read it seprately
        if counter!=3:                # 3 because we only need top 3 actors from the entire list
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[34]:


movies['cast'] = movies['cast'].apply(convertcrew)


# In[35]:


movies.head()


# In[38]:


def fetch_director(obj):                  #this function will pick the director's name from the entire list of crew and store them into "L"
    L=[]
    for i in ast.literal_eval(obj):      #ast.literal_eval coverts the input(obj) which is a string into a list so that we can read it seprately
        if i['job']== 'Director':
            L.append(i['name'])
            break
    return L


# In[40]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[41]:


movies.head()


# In[42]:


movies['overview'][0]


# In[44]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())  #to conver the string format of "overview" into a list


# In[45]:


movies.head()


# In[48]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])  #to remove the space between words so that we can convert them into hashtags and use them for searching similar tags
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])   


# In[49]:


movies.head()


# In[50]:


#create a new column named "tags" in which we will concatinate all the columns above which will give us tags similar to hasgtags

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()


# In[51]:


movies['tags'][0]


# In[54]:


new_df = movies[['movie_id','title','tags']] #create a new df/table which has only 3 columns from the movies df


# In[55]:


new_df


# new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))  #convert the list back into string

# In[61]:


new_df.head()


# In[63]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) # convert the strings into lower case as it is usually recomended 


# # MODEL BUILDING 

# We will convert these tags into vectors and then recomend movies based on the closest vectors.
# Converting the strings will require us to find 2000-5000 most occuring words in the entire column across all the rows and then 
# weigh each row entry based on how many times a particular word from the 5000 words list has occured in that row.
# For this we will use the vectorizer class in scikit learn library.
# Also we will be using bag of words and remove the stop words for better results.
# 

# In[87]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[88]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[73]:


vectors[0]


# In[89]:


cv.get_feature_names()


# In[90]:


#apply stemming to all the words in tags colummn
import nltk
from nltk.stem.porter import PorterStemmer

ps= PorterStemmer()


# In[91]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[92]:


new_df['tags'] = new_df['tags'].apply(stem)


# We will use cosine distance in place of Eulcedian distance due to high dimensionality

# In[94]:


from sklearn.metrics.pairwise import cosine_similarity


# In[97]:


similarity = cosine_similarity(vectors)


# Now we create a function which gives 5 similar movies when given an input

# In[100]:


def recomend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[103]:


recomend('Batman Begins')


# In[ ]:




