# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# Imports necessary libraries
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
from gensim import models, corpora, similarities
from gensim.parsing.preprocessing import preprocess_documents
import plotly.graph_objects as go

def loadRedditData():
    # Loads all saved Reddit posts
    df = pd.read_csv("data/reddit_todayilearned.csv")
    # Selects only the following columns
    df = df[["id", "author", "domain", "url",
             "num_comments", "score", "title",
             "retrieved_on", "over_18", "permalink",
             "created_utc", "link_flair_text"]]
    # Leaves only the non-adult content
    df = df[~df["over_18"]]
    # Removes documents with lower than 10 score
    df = df[df["score"] > 10]
    # Resets the index
    df.reset_index(inplace=True, drop=True)
    # Creates a list of documents
    documents = df["title"].tolist()
    # Preprocesses the documents
    texts = preprocess_documents(documents)
    # Creates the dictionary
    dictionary = corpora.Dictionary(texts)
    # Creates the corpus using bag-of-words
    corpus = [dictionary.doc2bow(text) for text in texts]
    # Generates the TF-IDF model
    tfidf = models.TfidfModel(corpus)
    # Creates the TF-IDF corpus
    corpus_tfidf = tfidf[corpus]
    # Fits an LSI model (with 100 topics)
    model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=15)
    # Identifies topics for each document
    corpus_wrapper = model[corpus_tfidf]
    # Creates the similarity index
    index = similarities.MatrixSimilarity(corpus_wrapper)
    return corpus_wrapper, index, df

def nextTIL():
  # Creates a set of documents the user has seen
  seen = [x[0] for x in user_actions]

  if len(seen) > 0:
      # Retrieves topic values for each document multiplied by the user action (1 or -1)
      user_topics = [[x[1]*user_action[1] for x in corpus_wrapper[user_action[0]]] for user_action in user_actions]
      # Computes the mean of topic values for each document
      user_profile = [(i, x) for i, x in enumerate(np.array(user_topics).mean(axis=0))]
  else:
    user_profile = [(x, 0) for x in range(len(corpus_wrapper[0]))]

  # The more actions the user takes, the less we Explore and stop at 10% Exploration rate
  diminishingExplore = lambda x: max(80 - 10 * np.log(x), 10) * 0.01

  # If no user actions have been taken we Explore
  if len(seen) == 0:
    explore = True
  # According to the diminishingExplore function
  elif np.random.uniform(low=0.0, high=1.0) < diminishingExplore(len(seen)):
    explore = True
  # The rest eighty percent of the time we Exploit
  else:
    explore = False

  # If we are Exploring – returns a random document from the Top 50 scoring unseen documents
  if explore == True:
    doc_idx = df[~df.index.isin(seen)].sort_values(by="score", ascending=False).head(50).sample(1).index[0]
    # Determines the topic with the highest value
    topic_num = np.array(corpus_wrapper[doc_idx])[:,1].argmax() + 1
    return doc_idx, explore, topic_num, user_profile, 0

  # Finds similarities between the user and the documents
  sim = index[user_profile]
  # Calculates the similarities weighted by the root of score of each document (from Reddit)
  w_sim = np.array(sim * np.power(df["score"], 0.03))
  # Sorts the weighted similarities
  w_sim_sorted_desc = w_sim.argsort()[::-1]
  # Removes seen documents from the array
  w_sim_sorted_desc_not_seen = np.delete(w_sim_sorted_desc, np.isin(w_sim_sorted_desc, seen))
  # Index of the top document
  doc_idx = w_sim_sorted_desc_not_seen[0]
  # Determines the topic with the highest value
  topic_num = np.array(corpus_wrapper[doc_idx])[:,1].argmax() + 1
  # Retrieves the user & document similarity
  user_doc_sim = np.round(sim[doc_idx], 2)
  return doc_idx, explore, topic_num, user_profile, user_doc_sim

# Sets the stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initializes the Dash App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Defines the App loyout
app.layout = html.Div([
    html.Div(dcc.Graph(id='topic-graph'),
             style={"width": "100%", "height":" 400px", "align-items": "right", "justify-content": "center", "display": "flex"}),
    html.Div(id="TIL", children="", style={"width": "800px","height": "150px", "margin": "auto", "text-align": "center", "font-size": "18px", "padding": "10px"}),
    html.Div([dcc.Link(id="TIL-url", href=""),
              " / ",
              dcc.Link(id="TIL-permalink", href="")],
             style={"width": "800px","height": "50px", "margin": "auto", "text-align": "center", "font-size": "18px", "padding": "10px"}),
    html.Div(id="TIL-topic", children="", style={"width": "800px", "margin": "auto", "text-align": "left", "font-size": "14px", "font-weight": "bold"}),
    html.Div(id="TIL-explore", children="", style={"width": "800px", "margin": "auto", "text-align": "left", "font-size": "14px", "font-weight": "bold"}),
    html.Div([html.Button('Upvote 👍', id='up_btn', n_clicks=0, style={"width": "200px"}),
              html.Button('Downvote 👎', id='down_btn', n_clicks=0, style={"width": "200px"})],
             style={"width": "100%", "height":" 100px", "align-items": "center", "justify-content": "center", "display": "flex"})
])

# Python functions that are automatically called by Dash whenever an input component's property changes
@app.callback(
    Output('TIL', 'children'),
    Output('TIL-url', 'children'),
    Output('TIL-permalink', 'children'),
    Output('TIL-topic', 'children'),
    Output('TIL-explore', 'children'),
    Output('topic-graph', 'figure'),
    Input('up_btn','n_clicks'),
    Input('down_btn','n_clicks'))


def displayNext(upvt, dnvt):
    # Defines upvotes and downvotes
    global TIL_id
    global user_actions
    # Checks if a button has been pressed
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    upvote = None
    # If the upvote button was pressed adds a user action with a value 1
    if "up_btn" in changed_id:
        user_actions.append((TIL_id, 1))
    # If the downvote button was pressed adds a user action with a value -1
    elif "down_btn" in changed_id:
        user_actions.append((TIL_id, -1))
    else:
        upvote = None

    # Retrieves the next TIL post
    TIL_id, TIL_explore, TIL_topic, user_profile, user_doc_sim = nextTIL()

    # Retrieves the next TIL post text
    TIL_text = df["title"].iloc[TIL_id]

    # Generates the TIL Topic Text
    TIL_topic_text = "Main Topic # is " + str(TIL_topic)

    # Generates text describing whether the recommendation is a popular factoid or a personalized factoid
    TIL_explore_text = "Personalized factoid, Match Score is " + str(user_doc_sim) if TIL_explore == False else "Popular factoid"
    # Adds the Reddit Score
    TIL_explore_text += ", Reddit Score is " + f'{df["score"].iloc[TIL_id]:,}'

    # Generates the URL link for the source
    TIL_url = dcc.Link("Source", href=df["url"].iloc[TIL_id])

    # Generates the URL link for the Reddit Post (permalink)
    TIL_permalink = dcc.Link("Reddit Post", href="https://www.reddit.com/" + df["permalink"].iloc[TIL_id])

    # Pulls out the first elements (indices) and adds 1 so that the topics indices start from 1
    TIL_topics = [x[0]+1 for x in user_profile]
    # Pulls out the second elements (topic scores)
    TIL_topic_scores = [x[1] for x in user_profile]

    # Generates the barplot containing Topic indices and scores
    fig = go.Figure(go.Bar(
        y=TIL_topics,
        x=TIL_topic_scores,
        orientation="h"))
    fig_x_range_max = np.max(np.abs(TIL_topic_scores))
    fig_x_range_max *= 1.2
    fig.update_layout(
        autosize=False,
        width=800,
        height=400,
        title=go.layout.Title(text="Your Topic Preferences"),
        xaxis={"range":[-fig_x_range_max, fig_x_range_max], "visible": False, "showticklabels": False},
        yaxis={"visible": True, "showticklabels": True, "tickvals": TIL_topics, "ticktext": ["Topic #" + str(x).zfill(2) for x in TIL_topics]}
    )

    return [TIL_text, TIL_url, TIL_permalink, TIL_topic_text, TIL_explore_text, fig]

# Loads Reddit data
corpus_wrapper, index, df = loadRedditData()
# Initializes user actions list
user_actions = []
# Initializes TIL_id
TIL_id = None

if __name__ == "__main__":
    # Runs the app
    app.run_server(debug=True)