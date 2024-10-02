import numpy as np
import tensorflow as tf
import pathlib
import sklearn.metrics

def _main():
    idg = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0
	# 学習時のスケールと合わせる必要がある
    )
    gen_validation = idg.flow_from_directory(
        directory='val',
        target_size=(64, 64),
        batch_size=16,
        shuffle=False
    )

    # 学習済みのモデルをロード
    model = tf.keras.models.load_model('model_compe_mobilenet_train.h5')
    preds = model.predict_generator(gen_validation)

    # 検証データは以下の15クラスが6枚ずつ
    labels = [
        '01_stop',
        '02_dontEntry',
        '03_max_20',
        '04_max_30',
        '05_max_40',
        '06_max_50',
        '07_donotParking',
        '08_donotParkingStopping',
        '09_pedestrian',
        '10_bicycleAndPedestrianOnly',
        '11_pedestrianOnly',
        '12_oneWay',
        '13_goOnlyinDirectionOfArrow',
        '14_schoolRoad',
        '15_end'
    ]
    # 正解ラベルの作成
    y_label = []
    for label in labels:
        for i in range(6):
            y_label.append(label)

    # 推論ラベルの作成
    y_preds = []
    for p in preds:
        # 確信度が最大の値を推論結果とみなす
        label_idx = p.argmax()
        y_preds.append(labels[label_idx])
    
    # 混合行列を取得
    val_mat = sklearn.metrics.confusion_matrix(y_label, y_preds, labels=labels)
    rec_score = sklearn.metrics.recall_score(y_label, y_preds, average=None)
    print('再現率： ',rec_score)

    pre_score = sklearn.metrics.precision_score(y_label, y_preds, average=None)
    print('適合率： ', pre_score)
    print()

    f1_score = sklearn.metrics.f1_score(y_label, y_preds, average=None)
    print('F値   ： ', f1_score)

    rec_score_avg = sklearn.metrics.recall_score(y_label, y_preds, average="macro")
    print('再現率(平均)： ', rec_score_avg)
    pre_score_avg = sklearn.metrics.precision_score(y_label, y_preds, average="macro")
    print('適合率(平均)： ', pre_score_avg)
    f1_score_avg = sklearn.metrics.f1_score(y_label, y_preds, average="macro")
    print('F値(平均)   ： ', f1_score_avg)
    print()

    acc_score = sklearn.metrics.accuracy_score(y_label, y_preds)
    print('正解率： ', acc_score)

if __name__ == '__main__':
    _main()
