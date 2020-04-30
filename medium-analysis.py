import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/medium-articles-tagged-in-mldlai/medium.csv')
data.head(3)
sns.set_color_codes("pastel")
tag = data.groupby('1.Tag').size()
top = list(tag.index)
value = list(tag.values)
sns.barplot(x=top, y=value, data=data,color="b")
for i in range(0,len(data)):
    if 'K' in data["5.Upvotes"][i]:
        data["5.Upvotes"][i]= int(float(data["5.Upvotes"][i].replace("K",""))*1000)
    else:
        data["5.Upvotes"][i] = int(data["5.Upvotes"][i])

data.head()
data = data.drop("Unnamed: 0",axis=1)
upvote=data.groupby("2.Name").sum()
upvote.head()
fig_dims=(50,15)
fig, ax = plt.subplots(figsize=fig_dims)
name = data.groupby('2.Name').size()
top = list(name.index)
value = list(name.values)
sns.barplot(x=top, y=value, data=data,ax=ax)
plt.xticks(rotation=90)
len(upvote.index)
upvote=(upvote[(upvote['5.Upvotes'] > 7000)])
plt.figure(figsize=(15,6))
plt.xticks(rotation='vertical')
sns.barplot(upvote.index, upvote["5.Upvotes"], alpha=0.8)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Number of Upvotes', fontsize=14)
plt.show()
name = data.groupby('2.Name').size()
name = name[(name.values>2)]
plt.figure(figsize=(15,5))
plt.xticks(rotation='vertical')
sns.barplot(name.index, name.values, alpha=0.8)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Number of Articles', fontsize=14)
plt.show()
data['len_text'] = data['4.Body'].str.len()
print(data['len_text'])
data['len_title'] = data['3.Title'].str.len()
sns.lmplot('len_title', 'len_text', data=data,order=3)
plt.show()

a4_dims = (35, 6)
fig, ax = plt.subplots(figsize=a4_dims)
sns.pointplot('len_text', '5.Upvotes', data=data)
plt.show()

a4_dims = (35, 10)

fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('2.Name', '5.Upvotes', data = data)
plt.xticks(rotation='vertical')

data['3.Title']= data['3.Title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['3.Title'].head()

 data['3.Title'] = data['3.Title'].str.replace('[^\w\s]','')
data['3.Title'].head()

from nltk.corpus import stopwords
stop = stopwords.words('english')
data['3.Title'] = data['3.Title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

def get_words_count(data_series, col):
    words_count = {}
    m = data_series.shape[0]
    for i in range(m):
        words = data[col].iat[i].split()
        for word in words:
            if word.lower() in words_count:
                words_count[word.lower()] += 1
            else:
                words_count[word.lower()] = 1
    return words_count

title_words = get_words_count(data, '3.Title')
title_words = pd.DataFrame(list(title_words.items()), columns=['words', 'count'])

title_words.sort_values(by='count', ascending=False).head(15)
from wordcloud import WordCloud
fig = plt.figure(dpi=100)
a4_dims = (12, 20)
fig, ax = plt.subplots(figsize=a4_dims)
wordcloud = WordCloud(background_color ='white', max_words=200,max_font_size=40,random_state=3).generate(str(title_words.sort_values(by='count', ascending=False)['words'].values[:20]))
plt.imshow(wordcloud)
plt.title = 'Top Word in the title of Medium Articles'
plt.show()
title_words.head()


topten_title_words = title_words.sort_values(by='count', ascending=False)['words'].values[:10]
data['topten_title_count'] = data['3.Title'].apply(lambda s: sum(s.count(topten_title_words[i]) for  i in range(10)))
data.head()

sns.set_color_codes("colorblind")
a4_dims = (12, 7)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('topten_title_count', '5.Upvotes', data = data,alpha=0.8,ci=None,color='c')

for i in range(0,len(data)):
    data['6.Date'][i] =''.join([i for i in data['6.Date'][i] if not i.isdigit()])
month = data.groupby('6.Date').size()
unique_months=month.index.unique()
print(unique_months)

plt.figure(figsize=(15,5))
plt.xticks(rotation='vertical')
sns.pointplot(unique_months, month.values)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Articles', fontsize=14)
plt.title("Articles with respect to the Month it was written", fontsize=20)
plt.show()
data.head()



uop = data.groupby(['6.Date'])['5.Upvotes'].sum()
print(uop)

sns.set_color_codes("pastel")
plt.figure(figsize=(15,5))
plt.xticks(rotation='vertical')
sns.barplot(uop.index, uop,data=data)