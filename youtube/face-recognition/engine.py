# preprocess image
import cv2
import numpy as np
from openvino.inference_engine import IECore
import os
from cvzone.FaceDetectionModule import FaceDetector


model_xml = 'public/face-recognition-resnet100-arcface-onnx/FP16/face-recognition-resnet100-arcface-onnx.xml'
model_bin = 'public/face-recognition-resnet100-arcface-onnx/FP16/face-recognition-resnet100-arcface-onnx.bin'

if os.path.exists(model_xml) and os.path.exists(model_bin):
    print('Model exists')
else:
    os.system("omz_downloader --name face-recognition-resnet100-arcface-onnx")
    os.system("omz_converter --name face-recognition-resnet100-arcface-onnx")
    print('Model downloaded')
    os.remove("public/face-recognition-resnet100-arcface-onnx/arcfaceresnet100-8.onnx")

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name='CPU')
input_name = next(iter(net.input_info))
input_shape = net.input_info[input_name].input_data.shape
out_name = next(iter(net.outputs))

face_detector = FaceDetector()


def preprocess(img, input_shape):
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = img/255.0
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = img.reshape(input_shape)
    return img


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unable to read image: %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def detect_and_crop_face(img):
    _, faces = face_detector.findFaces(img.copy())
    print(faces)
    if len(faces) == 1:
        x, y, w, h = faces[0]['bbox']
        face = img[y:y+h, x:x+w]
        return face
    else:
        return None


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_embeddings(img):
    img = detect_and_crop_face(img)
    if img is None:
        return None
    img = preprocess(img, input_shape)
    res = exec_net.infer(inputs={input_name: img})
    return res[out_name].reshape(-1)


def compare_embeddings(emb1, emb2, threshold=0.5):
    similarity = cosine_similarity(emb1, emb2)
    if similarity > threshold:
        return True, similarity
    else:
        return False, similarity