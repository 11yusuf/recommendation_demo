'''Web server that serves the recommender app.'''

import os
import random
from flask import Flask, request, render_template
import pickle
import random
import numpy as np 
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from sklearn import metrics

APP = Flask(__name__)

def get_max_recommended(arr):
    arr[arr <= 1.5] = 0
    recommended = []
    for i in range(10):
        maax = max(arr)
        index = arr.tolist().index(maax)
        recommended.append(index)
        arr[index] = 0
        
    return recommended

def get_predictions(user_id, model):
    pid_array = np.arange(53809, dtype=np.int32)
    uid_array = np.empty(53809, dtype=np.int32)
    uid_array.fill(user_id)
    predictions = model.predict(
            uid_array,
            pid_array,
            item_features=item_to_property_matrix_sparse,
            num_threads=4)
    
    return get_max_recommended(predictions)

@APP.route("/")
def index():
    '''Renderer for the root webpage.'''
    return render_template("index.html")

@APP.route("/recommend")
def recommend():
    '''Renderer for /recommend.'''
    # initial call to runplan which displays the planner simulation
    user_str = request.args.get('user', default=None)
    
    user = int(user_str)
    if user <= 0:
        return 'Please input a username'

    pred = get_predictions(user, loaded_model)

    return render_template("recommend.html", recommendations=pred, user=user)


if __name__ == "__main__":

    PORT = 8000
    item_to_property_matrix_sparse = sp.load_npz("item_to_property_matrix_sparse.npz")
    no_comp, lr, ep = 30, 0.01, 10
    loaded_model = pickle.load(open('savefile.pickle', 'rb'))
    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}/".format(PORT))
    # Set up the development server on port 8000.
    APP.debug = False
    APP.run()
