# Градиентный бустинг

* Скачать модель можно по [ссылке](https://disk.yandex.ru/d/Ryg5FBzU678f2g).

* Код чтение модели:

```python
import pickle
from catboost import CatBoostClassifier

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

# Дерево решений

* Скачать модель можно по [ссылке](https://disk.yandex.ru/d/gKmRMyEMaKKFKQ).

* Код чтение модели:

```python
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```
