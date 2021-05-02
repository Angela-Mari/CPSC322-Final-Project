# we are going to use Flask, a micro web framework
import os
import pickle 
from flask import Flask, jsonify, request 
import mysklearn.myutils as myutils
import sys

# make a Flask app
app = Flask(__name__)

# we need to add two routes (functions that handle requests)
# one for the homepage
@app.route("/", methods=["GET"])
def index():
    # return content and a status code
    return "<h1>Welcome to my App</h1>", 200

# one for the /predict 
@app.route("/predict", methods=["GET"])
def predict():
    # goal is to extract the 4 attribute values from query string
    # use the request.args dictionary
    gender = request.args.get("gender", "")
    region = request.args.get("region", "")
    highest_education = request.args.get("highest_education", "")
    imd_band = request.args.get("imd_band", "")
    age_band = request.args.get("age_band", "")
    num_of_prev_attempts = request.args.get("num_of_prev_attempts", "")
    studied_credits = int(request.args.get("studied_credits", ""))
    disability = request.args.get("disability", "")

    if "le" in age_band:
        age_band = age_band.split("le")
        age_band = age_band[0] + "<="
    if num_of_prev_attempts == "False" or num_of_prev_attempts == '0':
        num_of_prev_attempts = False
    elif num_of_prev_attempts == "True" or num_of_prev_attempts == '1':
        num_of_prev_attempts = True
    print("level:", gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability)
    prediction = predict_result_well([gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability])
    # if anything goes wrong, predict_interviews_well() is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else: 
        # failure!!
        return "Error making prediction", 400

def forest_predict(header, forest, instance):   
    y_predicted = []
    tree_predictions = []
    gen_header = []
    for item in header:
        gen_header.append("att" + str(header.index(item)))
    for tree in forest:
        tree_predictions.append(tdidt_predict(gen_header, tree, instance))
    y_predicted = majority_vote_predictions(tree_predictions)
    print(y_predicted)
    return y_predicted

def tdidt_predict(header, tree, instance):
    """ recusrive call to find prediciton
        Args:
            header(list of str): the header for the tree
            tree(list of list): the tree to parse
            instance(list): the item which we are preicting the result for
        Returns: prediction 
    """
    # returns "True" or "False" if a leaf node is hit
    # None otherwise 
    info_type = tree[0]
    if info_type == "Attribute":
        # get the value of this attribute for the instance
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # recurse, we have a match!!
                return tdidt_predict(header, value_list[2], instance)
    else: # Leaf
        return tree[1] # label

def majority_vote_predictions(predicitons):
    """picks the majority choice 
        Args: partiton(list of list): current items that need a majority vote
        Returns: the majority choice
    """
    options = []
    votes = []
    for item in predicitons:
        if item in options:
            votes[options.index(item)] += 1
        else:
            options.append(item)
            votes.append(1)
    
    return options[votes.index(max(votes))]

def predict_result_well(instance):
    infile = open("forest.p", "rb")
    header, forest = pickle.load(infile)
    infile.close()
    
    try: 
        return forest_predict(header, forest, instance)# recursive function
    except:
        return None


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port) # TODO: set debug to False for production
    # by default, Flask runs on port 5000