import json
import random

with open('backend/data/furniture.json', 'r') as f:
    data = json.load(f)

# Styles
styles = ["Minimalist", "Modern", "Bohemian", "Scandinavian", "Industrial", "Traditional"]

new_items = [
    {"name": "Velvet Throw Pillow", "id_prefix": "pillow"},
    {"name": "Linen Couch Cushion", "id_prefix": "cushion"},
    {"name": "Large Area Rug", "id_prefix": "rug"},
    {"name": "Standing Vanity Mirror", "id_prefix": "mirror"},
    {"name": "Floating Bookshelf", "id_prefix": "shelf"},
    {"name": "Oak Dresser Drawer", "id_prefix": "drawer"},
]

count = 1000
for style in styles:
    for item in new_items:
        new_vec = [random.uniform(-30, 30) for _ in range(32)]
        data.append({
            "id": f"{item['id_prefix']}-{count}",
            "name": f"{style} {item['name']}",
            "style": style,
            "image_url": f"https://loremflickr.com/400/300/interior,{item['id_prefix']}?lock={count}",
            "feature_vector": new_vec
        })
        count += 1

with open('backend/data/furniture.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Added new items to furniture.json")
