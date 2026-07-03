https://github.com/user-attachments/assets/2cda84b8-adef-4a5f-92b9-4d93f9a49b40

# End-to-End Autonomous Driving Loop via Behavioral Cloning

A real-time, closed-loop Cyber-Physical System (CPS) that deploys a deep convolutional neural network based on the **NVIDIA Autonomous Car architecture** to pilot a vehicle within a Unity-based simulator. The pipeline bridges a native **PyTorch** inference engine with the simulation environment via a low-latency, full-duplex **Socket.io** telemetry server.

---

## Tech Stack

* **Deep Learning Framework:** PyTorch, Torchvision
* **Networking & Middleware:** Python-Socket.io, Flask, Eventlet
* **Data Processing:** NumPy, Pillow (PIL), Base64/BytesIO

---

## Architecture & Pipeline Dataflow

The system functions as a continuous feedback loop: the simulator captures real-time environment observations and streams them as base64 string buffers, which are decoded, processed, passed through the model, and returned as mechanical commands.

### 1. Vision Preprocessing Pipeline (`val_transforms`)
To maximize feature extraction consistency between training and live inference, incoming dashboard frames undergo a strict spatial and statistical transformation:
* **Region of Interest (ROI) Cropping:** Slices the frame to drop the horizon/sky (`top=60, height=80`), forcing the network to isolate its attention purely to lane markings and asphalt boundaries.
* **Resolution Scaling:** Resizes the arbitrary camera resolution down to a tight 66 x 200 x 3 matrix matching the NVIDIA network constraints.
* **Statistical Normalization:** Normalizes pixel distributions using ImageNet means (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) to standardize edge activations across fluctuating track lighting.

### 2. Neural Network Architecture
The network core maps structural visual layers directly to a singular regression output:
* **Convolutional Feature Extractors:** 5 sequential Strided 2D Convolution layers equipped with **ELU (Exponential Linear Unit)** activation functions to extract robust spatial representations (lane lines, boundaries, curve angles).
* **Fully Connected Regressor:** Flattens the feature map through a **Dropout (0.5)** layer to suppress overfitting, followed by 4 dense linear layers tapering down to a single floating-point scalar representing the calculated `steering_angle`.

### 3. Dynamic Control Mechanics
* **Velocity-Adaptive Throttle:** Calculated dynamically via 1.0 - (speed / speed_limit) up to a high-momentum threshold of **30 MPH**, optimizing vehicle kinetics through sweeping stretches.
* **Low-Speed Torque Recovery:** A low-speed override rule that forces the throttle to `0.5` whenever velocity falls below `1.5 MPH`, providing the required physical torque to keep the car from stalling on steep mountain hairpin inclines.

