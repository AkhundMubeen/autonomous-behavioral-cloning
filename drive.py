import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict

sio = socketio.Server()
app = Flask(__name__)

# THE SAME VAL TRANSFORMATIONS THAT WE DEFINED FOR TRAINING THE MODEL
val_transforms = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.crop(img, top=60, left=0, height=80, width=320)),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# THE SAME NVIDIA INSPIRED MODEL ARCHITECTURE THAT WE USED FOR MODEL TRAINING
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ELU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# LOADING THE BEST TRAINED WEIGHTS FROM OUR SAVED MODEL
device = torch.device('cpu')
state_dict = torch.load('best_model_2.pth', map_location=device)

# REFINING THE PATH BY STRIPPING THE 'module.' PREFIX AS I TRAINED THE MODEL IN DATA PARALLEL MODE
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_state_dict[key[7:]] = value  # Strip 'module.' prefix
    else:
        new_state_dict[key] = value

model = MyModel().to(device)
model.load_state_dict(new_state_dict)
model.eval()

# Telemetry Stream Handling
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])
        
        # Decodes the image and keeps it as a PIL Image for torchvision transforms
        image_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(image_string)))
        
        # Processes image and adds batch dimension [1, 3, 66, 200]
        image_tensor = val_transforms(image).unsqueeze(0).to(device)
        
        # Predicting steering angle using our trained model
        with torch.no_grad():
            steering_angle = model(image_tensor).item()
            
        speed_limit = 30
        throttle = 1.0 - speed / speed_limit
        
        # This prevents physical stalls on the mountain inclines
        if speed < 1.5:
            throttle = 0.5
        
        print(f"Model Command -> Steering: {steering_angle:.4f} | Throttle: {throttle:.2f} | Speed: {speed:.2f}")
        
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=sid)

@sio.on('connect')
def connect(sid, environ):
    print('Simulator connected successfully!')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)