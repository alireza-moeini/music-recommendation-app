import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_toy_data(n_users=200, n_items=1000, interactions=15000, seed=0):
    rng = np.random.RandomState(seed)
    user_ids = rng.randint(0, n_users, size=interactions)
    item_ids = rng.randint(0, n_items, size=interactions)

    df = pd.DataFrame({"user": user_ids, "item": item_ids, "rating": 1.0})

    df = df.drop_duplicates(subset=["user", "item"]).reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, random_state=seed)

    return train, test, n_users, n_items
