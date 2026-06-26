import joblib
import torch
import torch.nn as nn
from pathlib import Path


class FakeNewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "fake_news_model.pt"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


vectorizer = joblib.load(VECTORIZER_PATH)

model = FakeNewsClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

def predict_news(text: str):
    # Convert text to TF-IDF features
    text_vector = vectorizer.transform([text])

    # Convert to PyTorch tensor
    text_tensor = torch.tensor(text_vector.toarray(), dtype=torch.float32)

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(text_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    label = "Real News" if predicted.item() == 1 else "Fake News"

    return {
        "prediction": label,
        "confidence": round(confidence.item() * 100, 2)
    }