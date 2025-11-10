from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np

# Model definition (same as your model.py CNN_LSTM)
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=1, num_features=28, lstm_hidden_size=64, lstm_layers=1, num_classes=15, dropout=0.5):
        super(CNN_LSTM, self).__init__()

        # CNN layers with fixed sizes matching the saved model
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM with single layer
        self.lstm = nn.LSTM(
            input_size=64,  # Match conv2 output channels
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        # Single fc layer as in saved model
        self.fc = nn.Linear(lstm_hidden_size, num_classes)  # Not fc1/fc2, just fc

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)  # Single fc layer
        return out
# Load model (CPU or CUDA based on availability)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set num_features to match input dimension (78 features)
num_features = 78
model = CNN_LSTM(
    input_channels=1,
    num_features=num_features,  # Number of input features from the data
    lstm_hidden_size=64,        # Match the saved model
    lstm_layers=1,             # Single layer LSTM as in saved model
    num_classes=15,            # Number of attack classes
    dropout=0.5
).to(device)

# Now load the state dict
model.load_state_dict(torch.load('Models/cnn_bilstm_best.pth', map_location=device))
model.to(device)
model.eval()

attack_classes = {
    0: "BENIGN",
    1: "Web Attack",
    2: "DoS",
    3: "DDoS",
    4: "PortScan",
    5: "Bot",
    6: "FTP-Patator",
    7: "SSH-Patator",
    8: "Infiltration",
    9: "Heartbleed",
    10: "Web Attack",
    11: "Brute Force",
    12: "Brute Force",
    13: "Brute Force",
    14: "Botnet"
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get('features', None)
    
    if features is None or not isinstance(features, list) or len(features) != 78:
        return jsonify({'error': 'Invalid input: must provide list of 78 features'}), 400
    
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        class_idx = np.argmax(probs)
        class_name = attack_classes.get(class_idx, "Unknown")
        
    response = {
        'predicted_class_idx': int(class_idx),
        'predicted_class_name': class_name,
        'confidence': float(probs[class_idx]),
        'all_class_probabilities': probs.tolist()
    }
    
    return jsonify(response)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

