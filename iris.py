import cv2
import insightface
from pymongo import MongoClient
import numpy as np

uri = "mongodb+srv://kazarani:kazarani@iris-cluster.mqcnbrp.mongodb.net/?retryWrites=true&w=majority&appName=iris-cluster"
client = MongoClient(uri)
db = client["iris-db"]
collection = db["embeddings"]
print("Connected to MongoDB ✅")

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=-1, det_size=(640, 640))

# rtsp_url = "rtsp://10.16.59.44:8554/live"
# rtsp_url = "rtsp://admin:admin@10.16.59.44:8554/live"
rtsp_url = "rtsp://localhost:8554/mystream"
capture = cv2.VideoCapture(rtsp_url)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


while True:
    gotFrame, frame = capture.read()
    if not gotFrame:
        print("❌ Did not get frame")
        break

    faces = model.get(frame)
    for face in faces:
        embedding = face.normed_embedding.astype(float).tolist()  # i embed here

        # compare mongodb and current embedding
        best_match = None
        best_score = -1

        for user in collection.find():
            db_embedding = user["embedding"]
            score = cosine_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_match = user

            # DRAW BOUNDING BOX
            if best_score > 0.6:
                print(
                    f"🟩 Recognized: {best_match['name']} ({best_match['roll_number']}) | Score: {best_score:.2f}"
                )
            else:
                print(f"🟥 Unknown face | Best score: {best_score:.2f}")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) != -1:
        break

capture.release()
cv2.destroyAllWindows()
