# CPSC322-Final-Project 🤖
This is Mia Brasil and Angela George's final data science project. The project uses a Random Forest Classifier to make predictions on if students from the input dataset will pass their online classes. The Random Forest Classifer was pickled and deployed on Heroku as an API that lets others input a query string to get a prediction for their student's parameters. 

## How to run 🧑‍💻
To run this project on your local machine, simply clone the repo and run the result_app.py file. 

Docker Note 🐳 : this project was run and deployed within a dokerized container. To install the exact enviornment for this project, run one of these commands:  

On Mac/Linux: docker run -i -t -p 8888:8888 -v "$PWD":/home --name anaconda3_cpsc322 continuumio/anaconda3:2020.11
On Windows powershell: docker run -i -t -p 8888:8888 -v ${PWD}:/home --name anaconda3_cpsc322 continuumio/anaconda3:2020.11

And to install Tabulate run:   
pip install tabulate

For deployment to your own Heroku Dynamo run these commands:  

1. If you have not already done so, make your top-level project directory into a local git repository (e.g. git init, git add -A, git commit -m "initial")
1. In your project's top-level directory, create a Heroku app from your project: heroku create <app name>
1. This should add a new remote to your local Git repo for Heroku (confirm with git remote -v)
1. Set the stack of your app to container: heroku stack:set container
1. Push your app to Heroku: git push heroku master

Docker Note 🐳 : We did not modify the continuumio/anaconda3 image that was pushed to heroku. Thus the tabulate library will not work when pushing to heroku. Our solution was to comment out calls and imports of tabulte since they are not essential for the API but an alternative solution would be to add that import in the Docker file that is used to build the Docker image on Heroku. 

Docker commands, heroku file structure, and heroku commands compiled by Gina Sprint. 

To get a prediction here is an example query string:    
[your url]/predict?gender=M&region=East+Anglian+Region&highest_education=HE+Qualification&imd_band=90-100%&age_band=55le&num_of_prev_attempts=False&studied_credits=1&disability=N

## Organization 📂

This project is organized into: 

* input_data which was the data we got from [Open University Learning Analytics](https://analyse.kmi.open.ac.uk/open_dataset)
* mysklearn which is a folder of from scratch evaluators and classifers based on Scikit learn.
* test_myrandomforestclassifier.py which tests the classifier

* a Technical Report Jupyter Notebook which visualizes data and findings
* python (flask) and heroku files for deployment

