# --- coding: utf-8 --

#Recommender system for movies using collaborative filtering
#Mio Terada (Jan. 22, 2017)

import numpy as np
import pandas as pd
import scipy.sparse as sp



def als4ml100k(v1=20):
    """ Matrix factorization with ALS for collaborative filtering.
    It return matrix (W dot H^T).
    This script is for the exclusive use of MovieLens 100k Dataset.
    v1: number of factors. It should be integer. Default is 20. """

    #--- Read Data ---#
    df = pd.read_csv('ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])

    #--- Create Matrix ---#
    user_sz,item_sz = (df.max().ix['user_id'],df.max().ix['item_id'])
    X = sp.lil_matrix((user_sz,item_sz))
    
    for i in df.index:
        values = df.ix[i]
        X[values['user_id']-1,values['item_id']-1] = values['rating']

    fct = v1 #number of factors is set to v1
    W = np.zeros((user_sz,fct)) #matrix for user
    H = np.zeros((item_sz,fct)) #matrix for items

    #--- Solve X=W*H using ALS ---#
    H[:,0] = X.mean(axis=0) #set initial values
    H[:,1:] = np.random.rand(item_sz,fct-1) #set initial values
    
    I = np.eye(v1)
    lmd = 0.01 #regulation parameter

    err = 1.e+9
    err0 = 1.e+10
    while err<err0-0.01: #Repeat until convergence
        err0 = err

        #LS for user
        HH = np.dot(H.T,H)
        HHH = np.dot(np.linalg.inv(HH+lmd*I),H.T)
        for i in range(user_sz):
            coef = np.dot(HHH,X[i,:].T.todense())
            W[i,:] = coef.T

        #LS for items
        WW = np.dot(W.T,W)
        WWW = np.dot(np.linalg.inv(WW+lmd*I),W.T)
        for j in range(item_sz):
            coef = np.dot(WWW,X[:,j].todense())
            H[j,:] = coef.T
        
        err = np.nansum(np.power((X-np.dot(W,H.T)),2))
        print(err)

    #--- Save matrix ---#
    rates = np.dot(W,H.T)
    wfile = './rates_ALS_lmd0.1.bin'
    with open(wfile, "w") as f:
        rates = np.array(rates,dtype=np.float32)
        rates.tofile(f)
    
    return np.dot(W,H.T)




def recommend(v1, v2=3, v3=0, v4=20):
    """ Return the recommended movie ID and title for specified user.
    v1, v2, v3 and v4 must be integer.
    v1: user id.
    v2: number of movies you want to recommend. Default is 3.
    v3: If v3=0, read existing matrix. If v3=1, renew matrix. Default is 0.
    v4: number of factors for matrix factorization. It should be integer. Default is 20.  """

    #--- Read original data ---#
    df = pd.read_csv('ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
    user_sz,item_sz = (df.max().ix['user_id'],df.max().ix['item_id'])
    org_rate = df[df.user_id==v1]   
    dfitem = pd.read_csv('ml-100k/u.item',sep='|',names=['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy',\
'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'])

    #--- Read matrix ---#
    if v3==0:
        rfile = "./rates_ALS.bin"
        vtl_mtrx = np.fromfile(rfile,'float32')
        vtl_mtrx = vtl_mtrx.reshape(user_sz,item_sz)
    else:
        from recsystem import als4ml100k
        vtl_mtrx = als4ml100k(v4)

    vtl_rate = vtl_mtrx[v1-1,:]
    vtl_rate[org_rate.item_id-1]=0 #rated movies should be removed
    
    #--- Get item_id of higher rank ---#
    vtl_df = pd.DataFrame({'item_id':np.arange(1,item_sz+1), 'rates':vtl_rate})
    recitems=vtl_df.sort_index(by='rates',ascending=False)[:v2].item_id
    recitems=recitems.values

    #--- Return movie title ---#
    print('Recommended Movies for ID %d' % (v1))
    for i in range(v2):
        print('Title: %s' % (dfitem.movie_title[recitems[i]-1]))


    
    
