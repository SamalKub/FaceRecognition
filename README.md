# FaceRecognition
Training MobNet on CASIA-WEB-FACE with Softmax Loss &amp; Center loss and Validating Models on LFW with PyTorch

## Data
Обучающий датасет: CASIA WebFace https://arxiv.org/pdf/1411.7923.pdf , 10572 классов (уникальных лиц), 490623 изображений
Тестовый датасет: LFW http://vis-www.cs.umass.edu/lfw/, 3000 positive и 3000 negative пар.

# Installation
First of all, you should have python 3.x to work with this project. The recommended Python version is 3.6 or greater.

Note for Windows users: You should start a command line with administrator's privileges.

First of all, clone the repository:
```
git clone https://github.com/SamalKub/FaceRecognition.git
cd face_recognition/
```
Create a new virtual environment:

    # on Linux:
    python -m venv facerecvenv
    # on Windows:
    python -m venv facerecvenv

Activate the environment:

    # on Linux:
    source facerecvenv/bin/activate
    # on Windows:
    call facerecvenv\Scripts\activate.bat

Install required dependencies:

    # on Linux:
    pip install -r requirements.txt
    # on Windows:
    python -m pip install -r requirements.txt

## Models checkpoints
All checkpoints for models you can find here: https://drive.google.com/drive/folders/19SMu--m3L9Ot89Dpg5Q3mCFVA1MO2yIA?usp=drive_link
