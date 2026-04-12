import json
import random

with open('backend/data/furniture.json', 'r') as f:
    data = json.load(f)

# Styles
styles = ["Minimalist", "Modern", "Bohemian", "Scandinavian", "Industrial", "Traditional"]

new_items = [
    {"name": "Ceramic Table Vase", "id_prefix": "vase"},
    {"name": "Monstera Potted Plant", "id_prefix": "plant"},
    {"name": "Abstract Canvas Decor", "id_prefix": "decor"},
    {"name": "Wall Clock", "id_prefix": "clock"},
    {"name": "Woven Storage Basket", "id_prefix": "basket"},
    {"name": "Oak TV Stand", "id_prefix": "tv"},
]

count = 0
for style in styles:
    for item in new_items:
        new_vec = [random.uniform(-30, 30) for _ in range(32)]
        data.append({
            "id": f"{item['id_prefix']}-{count}",
            "name": f"{style} {item['name']}",
            "style": style,
            "image_url": f"https://picsum.photos/seed/{item['id_prefix']}{count}/400/300",
            "feature_vector": new_vec
        })
        count += 1

with open('backend/data/furniture.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Added items to furniture.json")
