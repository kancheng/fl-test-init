# PKL Note

1. data unzip

```
bash unzip.sh
```

2. run python files.

```
python t.py
```

3. code.

```
import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
print(len(data))
print(type(data))
```

## Reference

- https://blog.csdn.net/taotiezhengfeng/article/details/78148529

- https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file


