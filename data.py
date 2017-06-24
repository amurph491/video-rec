import pandas as pd
import collections
import implicit
from scipy.sparse import lil_matrix


def import_data():
    df_user_table = pd.read_csv("User_Table.csv", header=0)
    df_video_table = pd.read_csv('Video_table.csv', index_col=0, header=0)
    df_user_video = pd.read_csv('User_Video.csv', index_col=0, header=0)

    favs = df_user_table.favorites.map(lambda f: f.split(sep=';'))
    count = favs.apply(collections.Counter)
    be_favs = pd.DataFrame.from_records(count).fillna(value=0)
    df_ups = pd.concat([df_user_table.user_id, be_favs], axis=1).set_index('user_id')
    df_vcat = pd.get_dummies(df_video_table, prefix='vcat')

    df_user_videos = df_user_video.join(df_vcat, how='left', on='video_id')
    df = df_ups.join(df_user_videos)
    return df

def create_matrix():
    df_user_table = pd.read_csv("User_Table.csv", header=0)
    df_video_table = pd.read_csv('Video_table.csv', index_col=0, header=0)
    df_user_video = pd.read_csv('User_Video.csv', header=0)

    favs = df_user_table.favorites.map(lambda f: f.split(sep=';'))
    count = favs.apply(collections.Counter)
    be_favs = pd.DataFrame.from_records(count).fillna(value=0)
    df_ups = pd.concat([df_user_table.user_id, be_favs], axis=1).set_index('user_id')
    df_vcat = pd.get_dummies(df_video_table, prefix='vcat')
    n_users = len(df_ups)
    n_videos = len(df_vcat)
    U_V = lil_matrix((n_users + 1, n_videos + 1))

    U_V[(df_user_video.user_id, df_user_video.video_id)] = 1.0
    # initialize a model
    U_V = U_V.tocsr()
    model = implicit.als.AlternatingLeastSquares(factors=50)

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(U_V.T)
    return model
