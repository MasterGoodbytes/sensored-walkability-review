import json

# Load data
data = json.load(open('data/01_scopus_results.json', encoding='utf-8'))

# Check affiliation structure
for i in range(5):
    if 'affiliation' in data[i] and data[i]['affiliation']:
        print(f"Article {i}:")
        print(f"Affiliation type: {type(data[i]['affiliation'])}")
        if isinstance(data[i]['affiliation'], list):
            print(f"First affiliation: {json.dumps(data[i]['affiliation'][0], indent=2)[:800]}")
        elif isinstance(data[i]['affiliation'], dict):
            print(f"Affiliation: {json.dumps(data[i]['affiliation'], indent=2)[:800]}")
        print("\n---\n")
        break
