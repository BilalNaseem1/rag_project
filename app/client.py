import requests

response = requests.post(
    "http://localhost:8000/search/invoke",
    json = {'input': {'question': "How is BTC going to perform after halving?"}})

print(response)