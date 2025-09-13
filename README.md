# Face_recognition_system
⚙️ Installation

Clone the repository

git clone https://github.com/durgaprasad18-k/AI-Powered-Face-Detection-Based-Attendance-System/tree/main
AI-Face-Attendance


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt


Set up MySQL database

Create a database attendance_system

Import attendance.sql (if provided)

Update DB credentials in app.py

▶️ Usage

Train the model

python train_model.py


Run the Flask server

python app.py


Open in browser

http://127.0.0.1:5000/


Upload face data & mark attendance automatically.
