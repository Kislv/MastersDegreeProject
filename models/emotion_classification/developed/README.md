# Градиентный бустинг

* Скачать модель можно по [ссылке](https://disk.360.yandex.ru/d/oqE4Vr0jXG52Og).

* Код чтение модели:

```python
import pickle
from catboost import CatBoostClassifier

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

# Дерево решений

* Скачать модель можно по [ссылке](https://disk.360.yandex.ru/d/SJ1hWRBmH_gW1A).

* Код чтение модели:

```python
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```
