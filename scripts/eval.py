from trainer import train_model
from data import generate_toy_data

train_df, test_df, n_users, n_items = generate_toy_data()
model = train_model(train_df, n_users, n_items, epochs=1)
print("Model trained for evaluation.")
