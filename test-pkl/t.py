import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
print(len(data))
print(type(data))

