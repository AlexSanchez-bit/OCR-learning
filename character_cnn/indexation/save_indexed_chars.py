import json

from constants import INDEX_MAP_PATH

#save indexed chars
def save_as_json(idx_to_char):
  with open(INDEX_MAP_PATH, "w") as f:
      json.dump(idx_to_char, f)

#load indexed chars
def load_from_json():
  with open(INDEX_MAP_PATH, "r") as f:
      idx_to_char= json.load(f)
  return idx_to_char
