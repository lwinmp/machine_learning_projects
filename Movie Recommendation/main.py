import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch the MovieLens dataset (filtering for ratings of 4.0 or higher)
data = fetch_movielens(min_rating=4.0)

# Print training and testing data sparse matrices to verify the dataset split
print(repr(data['train']))
print(repr(data['test']))

# Initialize the LightFM class using the WARP loss function
# WARP stands for Weighted Approximate Rank Pairwise and helps optimize recommendations
model = LightFM(loss='warp')

# Train the hybrid model on the training data over 30 epochs
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    # Number of users and movies in our dataset
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        # Get the movies this user already rated highly (known positives)
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Predict recommendations for all movies in the dataset
        scores = model.predict(user_id, np.arange(n_items))
        
        # Sort scores in descending order to get the highest recommendations first
        top_items = data['item_labels'][np.argsort(-scores)]

        # Print the results for each user
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")
        for x in top_items[:3]:
            print("        %s" % x)

# Call the function with model, data, and three sample user IDs
sample_recommendation(model, data, [3, 25, 450])