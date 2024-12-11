# %%
#find your information using username
CLIENT_ID = 'YourClientID'
SECRET_KEY = 'YourKey'

# %%

import requests
import pandas as pd
import json
import time

# %%
auth = requests.auth.HTTPBasicAuth(CLIENT_ID,SECRET_KEY)

# %%
data = {
    'grant_type': 'password',
    'username': 'YourUsername', #data collection username
    'password': 'YourPassword' #data collection password
}

# %%
headers = {'User-Agent': 'MyAPI/0.0.1'}

# %%
res = requests.post('https://www.reddit.com/api/v1/access_token',
                   auth = auth, data = data, headers=headers)

# %%
TOKEN = res.json()['access_token']

# %%
headers = {**headers, **{'Authorization': f'bearer {TOKEN}'}}

# %%
requests.get('https://oauth.reddit.com/api/v1/me', headers = {'User-Agent': 'MyAPI/0.0.1'})

# %%
requests.get('https://oauth.reddit.com/api/v1/me', headers = headers).json()

# %%
subreddit_names = [
    'subreddits' #enter subreddits seperated by commas
   
    
]

# %%
#pulls from hot posts of asked subreddits
post_all = []

def extract_comments(comments_data):
    comments_list = []

    if isinstance(comments_data, list):
        for item in comments_data:
            comments_list.extend(extract_comments(item))
    elif isinstance(comments_data, dict):
        if 'data' in comments_data and 'children' in comments_data['data']:
            for child in comments_data['data']['children']:
                comments_list.extend(extract_comments(child))
        elif 'body' in comments_data['data']:
            comment_info = {
                'author': comments_data['data']['author'],
                'body': comments_data['data']['body']
            }
            comments_list.append(comment_info)

    return comments_list

for subreddit_name in subreddit_names:
    url = f'https://oauth.reddit.com/r/{subreddit_name}/hot'
    response = requests.get(url, headers=headers, params={'limit': '100'})

    if response.status_code == 200:
        data = response.json()
        posts = data["data"]["children"]

        for post in posts:

            post_data = {
                'subreddit': post['data']['subreddit'],
                'post_id': post["data"]["id"],
                'Title': post['data']['title'],
                'SubTitle': post['data']['selftext'],
                'ups': post['data']['ups'],
                'downs': post['data']['downs'],
                'score': post['data']['score']
            }

            post_url = post['data']['url']
            post_id = post_url.split('/')[-3] 

            comments_url = f"https://www.reddit.com/comments/{post_id}.json"
            comments_response = requests.get(comments_url, headers={"User-Agent": "MyAPI/0.0.1"})
            print(comments_response.status_code)

            if comments_response.status_code == 200:
                comments_data = comments_response.json()

                comments = extract_comments(comments_data)

                post_data['comments'] = comments
            else:
                print(f"Failed to retrieve comments for post {post_data['post_id']}")
                print("Waiting 5 mins...")
                time.sleep(300)
            post_all.append(post_data)
    else:
        print(f"Failed to retrieve posts. Status code: {response.status_code}")

# %%
#prints column information
for post in post_all:
    print("Post ID:", post["post_id"])
    print("Subreddit:", post["subreddit"])
    print("SubTitle:", post["SubTitle"])
    print("Ups:", post["ups"])
    print("Downs:", post["downs"])
    print("Score:", post["score"])
    
    comments = post.get("comments", [])
    print("Comments:")
    for comment in comments:
        print("\nComment Author:", comment["author"])
        print("\nComment Body:", comment["body"])
    print("\n") 

# %%
#saves info to FetchedData
with open('FetchedData.json', 'w') as json_file:
    json.dump(post_all, json_file)

with open('FetchedData.json', 'r') as json_file:
    data = json.load(json_file)

# %%

df_new = pd.DataFrame(post_all)

# %%
#converts to CSV
df_new.to_csv('FetchedData.csv', index=False)

# %%
df_whatis = pd.json_normalize(data)

comments_column = []
for d in data:
    if 'comments' in d: 
        for c in d['comments']:
            c['post_id'] = d['post_id']
            comments_column.append(c)
    else:
        continue

comments_df = pd.json_normalize(comments_column)


df_whatis = df_whatis.merge(comments_df, how='left', on=['post_id'])


# %%
df_whatis

# %%
df_whatis.to_csv('FinalDataWithCommentAuthor&Body.csv', index=False)

# %%
column_to_drop = 'comments'
df_whatis_afterDrop = df_whatis.drop(column_to_drop, axis=1)

# %%
df_whatis_afterDrop

# %%
df_whatis_afterDrop.rename(columns={'subreddit': 'Subreddits'}, inplace=True)
df_whatis_afterDrop.rename(columns={'post_id': 'Post ID'}, inplace=True)
df_whatis_afterDrop.rename(columns={'ups': 'UpVotes'}, inplace=True)
df_whatis_afterDrop.rename(columns={'downs': 'DownVotes'}, inplace=True)
df_whatis_afterDrop.rename(columns={'score': 'Score'}, inplace=True)
df_whatis_afterDrop.rename(columns={'author': 'Comment Author'}, inplace=True)
df_whatis_afterDrop.rename(columns={'body': 'Comment Body'}, inplace=True)

# %%
df_whatis_afterDrop

# %%
df_whatis_afterDrop.to_csv('YourFinal.csv', index=False)

# %%
column_name = 'Comment Body'
empty_row_count = df_whatis_afterDrop[column_name].isna().sum()
print(f"Count of empty rows in column '{column_name}': {empty_row_count}")

# %%
##Do data preprocessing!!
##Data exploration!!

# %%


# %%



