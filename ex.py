import json

with open('yes24_best.json.', 'r',encoding='utf8') as f:
    json_data = json.load(f)

print(json_data)