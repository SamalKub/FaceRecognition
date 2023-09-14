configurations = {
    1: dict(
        SEED=13,
        LR_SOFTMAX=0.0001,
        FLAG_CENTER=False, #for center loss
        LR_CENTER=0.001, #for center loss
        ALPHA = 0.02,
        FEAT_DIM = 512,#256,512,512
        ALPHA_1 = 10,
        TRAIN_BATCH_SIZE=256,#256,
        VAL_BATCH_SIZE=1,
        NUM_EPOCHS=10,#90,
        WEIGHT_DECAY=0.0005,
        RGB_MEAN=[0.485, 0.456, 0.406],
        RGB_STD=[0.229, 0.224, 0.225],
        MODEL_NAME='MobNet', #'ResNet34', 'MobNet', 'InceptionResnetV1'
        TRAIN_PATH='/content/data/casia_faces',
        VAL_PATH='/content/data/lfw',
        PLOT_PATH='/content/gdrive/MyDrive/Samal_experiments/DL/FaceRecognition/plots',
        LOGS_PATH='/content/gdrive/MyDrive/Samal_experiments/DL/FaceRecognition/logs',
        FILE_EXT='.jpg',
        OPTIM='Adam',
        CHECKPOINT='/content/gdrive/MyDrive/Samal_experiments/DL/FaceRecognition/models',
    ),
}