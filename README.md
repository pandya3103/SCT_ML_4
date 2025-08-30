# Hand Gesture Recognition System

A real-time hand gesture recognition system using deep learning and computer vision. This project can identify various hand gestures through a webcam feed, enabling intuitive human-computer interaction.

## ✨ Features

- Real-time hand gesture recognition
- Multiple gesture support (palm, fist, thumb, index, etc.)
- High accuracy with deep learning model
- Simple and intuitive interface
- Cross-platform compatibility

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pandya3103/SCT_ML_4.git
   cd SCT_ML_4
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate   # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the main script:
   ```bash
   python basic_gesture.py
   ```

2. Position your hand in the green box on the screen
3. The system will detect and display the recognized gesture
4. Press 'q' to quit the application

## 📂 Project Structure

```
SCT_ML_4/
├── models/                  # Pre-trained models
│   └── hand_gesture_model.h5
├── basic_gesture.py         # Main application script
├── camera_test.py          # Webcam testing utility
├── gesture_recognition_webcam.py  # Alternative implementation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Dependencies

- OpenCV
- TensorFlow
- NumPy
- Matplotlib (for visualization)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue on GitHub.

---

Made with ❤️ by [Your Name]
