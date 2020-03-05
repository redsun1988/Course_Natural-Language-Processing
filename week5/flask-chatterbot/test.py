import csv
import io
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from IPython.core.debugger import set_trace
import os
import numpy as np
from utils import question_to_vec

with open('models\\word_embeddings.pkl', 'rb') as f:
    starspace_embeddings = pickle.load(f)
    embeddings_dim = 300


posts_df = pd.read_csv('models\\tagged_posts.tsv', sep='\t')
gPosts = posts_df.groupby("tag").count()
counts_by_tag = dict(zip(gPosts.index,gPosts.post_id))

for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]
    
    tag_post_ids = tag_posts.post_id.values
    titles = tag_posts.title.values
    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)

    for i, title in enumerate(titles):
        tag_vectors[i, :] = question_to_vec(title, starspace_embeddings, embeddings_dim)

    # Dump post ids and vectors to a file.
    filename = os.path.join("thread_embeddings_by_tags", os.path.normpath('%s.pkl' % tag))
    pickle.dump((tag_post_ids, titles, tag_vectors), open(filename, 'wb'))
'''
vectorizer = CountVectorizer(lowercase=True, stop_words="english", min_df=2)

dialogue_df = pd.read_csv('models\\dialogues.tsv', sep='\t')
posts_df = pd.read_csv('models\\tagged_posts.tsv', sep='\t')

corpus = list(dialogue_df['text'].values) + list(posts_df.title.values)

vectorizer.fit(corpus)
common_words = vectorizer.get_feature_names()


files = ["word_embeddings150000.tsv", "word_embeddings175000.tsv", "word_embeddings200000.tsv", "word_embeddings225000.tsv", "word_embeddings250000.tsv",
"word_embeddings25000.tsv", "word_embeddings275000.tsv", "word_embeddings300000.tsv", "word_embeddings325000.tsv", "word_embeddings350000.tsv",
"word_embeddings375000.tsv", "word_embeddings50000.tsv", "word_embeddings75000.tsv", "word_embeddings125000.tsv"
]

embeding = {}

for f_name in files:
    with io.open("models\\"+f_name, encoding='utf-8') as csvfile:
        f_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in f_reader:
            word = row[0]
            emb = row[1]
            if (word in common_words):
                print(word)
                embeding[word] = [float(item) for item in emb.replace("[", "").replace("]","").replace("\n","").split(" ") if item != ""]

with open("word_embeddings.tsv", mode="wb") as f:
    pickle.dump(embeding, f)

print(embeding)
print("++++++++++++++++++++++FINISH++++++++++++++++++++++")
'''