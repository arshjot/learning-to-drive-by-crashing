class Config:
    model_img_h, model_img_w = 128, 128

    val_split = 0.2
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.001
    lr_reduce_patience = 8
    es_patience = 20

    architecture = 'MobileNetV2'

    data_file = 'data.pkl'
