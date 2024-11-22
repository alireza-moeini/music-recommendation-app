import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MFModel
import os


def train_model(train_df, n_users, n_items, epochs=3, batch_size=1024, lr=1e-2, device="cpu"):
    users = torch.tensor(train_df["user"].values, dtype=torch.long)
    items = torch.tensor(train_df["item"].values, dtype=torch.long)
    labels = torch.tensor(train_df["rating"].values, dtype=torch.float32)

    dataset = TensorDataset(users, items, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MFModel(n_users, n_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for u, i, y in loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            pred = model(u, i)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * u.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataset):.4f}")

    # Save the model
    os.makedirs("models/model_store", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
        },
        "models/model_store/model.pt",
    )
    print("Model saved to models/model_store/model.pt")

    return model
