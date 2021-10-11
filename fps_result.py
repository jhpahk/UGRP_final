import gzip, pickle

with gzip.open("fps_hrnet.pkl", "rb") as f:
    fps_hrnet = pickle.load(f)

print(fps_hrnet)

with gzip.open("fps_first_prev.pkl", "rb") as f:
    fps_first_prev = pickle.load(f)

print(fps_first_prev)

with gzip.open("fps_only_prev.pkl", "rb") as f:
    fps_only_prev = pickle.load(f)

print(fps_only_prev)