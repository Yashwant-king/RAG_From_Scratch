import json

with open("run_files/run_files/result.json", "r") as f:
    data = json.load(f)
    print(data[0])
    print(type(data))  #  list
    