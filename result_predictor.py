import requests # lib to make http requests
import json # lib to help with parsing JSON objects

# url = "https://interview-flask-app.herokuapp.com/predict?level=Junior&lang=Java&tweets=yes&phd=yes"
url = "http://127.0.0.1:5000/predict?gender=M&region=East+Anglian+Region&highest_education=HE+Qualification&imd_band=90-100%&age_band=55le&num_of_prev_attempts=False&studied_credits=1&disability=N"


# make a GET request to get the search results back
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
response = requests.get(url=url)

# first thing... check the response status code 
status_code = response.status_code
print("status code:", status_code)

if status_code == 200:
    # success! grab the message body
    json_object = json.loads(response.text)
    print(json_object)