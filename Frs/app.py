import cv2
import os
from flask import Flask, request, render_template, send_file, redirect
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from time import sleep

app = Flask(__name__)

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load Haar cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create required directories
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Create today's CSV if not exists
csv_path = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        f.write('Name,Roll,Time\n')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces, labels = [], []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance(date):
    file_path = f'Attendance/Attendance-{date}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df['Name'].tolist(), df['Roll'].tolist(), df['Time'].tolist(), len(df)
    return [], [], [], 0

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in df['Roll'].tolist():
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')
        return True
    return False

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance(datetoday)
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, 
                           totalreg=totalreg(), datetoday=datetoday, datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html', mess='No trained model found. Please add a new face.', 
                               totalreg=totalreg(), datetoday2=datetoday2)

    cap = cv2.VideoCapture(0)
    frame_count = 0
    mess = "No face recognized."
    already_marked = set(pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')['Roll'].tolist())

    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            mess = "Failed to read from camera."
            break

        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            try:
                identified_person = identify_face(face.reshape(1, -1))[0]
                name, uid = identified_person.split('_')
                if int(uid) in already_marked:
                    mess = f"Attendance already taken for {name}."
                else:
                    add_attendance(identified_person)
                    mess = f"Attendance marked successfully for {name}."
                break
            except:
                mess = "Face not recognized."

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) == 27:
            mess = "Exited by user."
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance(datetoday)
    return render_template('index.html', names=names, mess=mess, rolls=rolls, times=times, l=l, 
                           totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add_user', methods=['POST'])
def add_user():
    newusername = request.form['name']
    newuserid = request.form['user_id']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    os.makedirs(userimagefolder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i = 0

    while i < 20:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.imwrite(f'{userimagefolder}/{i}.jpg', face_img)
            i += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Images Captured: {i}/20", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Capturing Images - Press Q to Quit", frame)
            sleep(1)
            break
        else:
            cv2.putText(frame, f"Please position your face! Captured: {i}/20", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Capturing Images - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    train_model()
    names, rolls, times, l = extract_attendance(datetoday)
    mess = f"User {newusername} added and model trained successfully."
    return render_template('index.html', mess=mess, names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/download_csv/<date>')
def download_csv(date):
    filepath = f'Attendance/Attendance-{date}.csv'
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404

@app.route('/users')
def view_users():
    users = []
    for folder in os.listdir('static/faces'):
        if '_' in folder:
            name, roll = folder.split('_')
            users.append({'name': name, 'roll': roll, 'folder': folder})
    return render_template('users.html', users=users)

@app.route('/delete_user/<folder>')
def delete_user(folder):
    path = os.path.join('static/faces', folder)
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
        train_model()
        return redirect('/users')
    return "User not found", 404

@app.route('/retrain')
def retrain():
    train_model()
    return "Model retrained successfully"

if __name__ == '__main__':
    app.run(debug=True)
