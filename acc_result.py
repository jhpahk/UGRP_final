import gzip
import pickle

with gzip.open("acc_first_prev.pkl", "rb") as f:
    acc_first_prev = pickle.load(f)

print(acc_first_prev)

with gzip.open("acc_only_prev.pkl", "rb") as f:
    acc_prev_only = pickle.load(f)

print(acc_prev_only)