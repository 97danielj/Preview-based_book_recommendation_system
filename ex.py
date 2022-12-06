import json

with open('preview_data/yes24_steady.json', 'r',encoding='utf8') as f:
    json_data = json.load(f)

print(json_data)