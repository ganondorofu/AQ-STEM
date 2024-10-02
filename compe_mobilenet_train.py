import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_result(history):
    '''
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    '''

    # accuracy
    plt.figure()
    plt.plot(history.history['acc'], label='acc', marker='.')
    plt.plot(history.history['val_acc'], label='val_acc', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('accuracy')
    fig_name = os.path.splitext(os.path.basename(__file__))[0] + '_accuracy.png'
    plt.savefig(fig_name)
    # plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.plot(history.history['val_loss'], label='val_loss', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    fig_name = os.path.splitext(os.path.basename(__file__))[0] + '_loss.png'
    plt.savefig(fig_name)
    # plt.show()

def create_model_mobilenet(num_classes, input_shape):
    """
    モデル作成(Mobilenet)
        
    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : tuple
        入力画像サイズ
        (32, 32, channels)以上の指定が必要
    """
    # ImageNetで事前学習したMobileNetモデルを読込
    base_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                           include_top=False,
                                                           input_shape=input_shape)
    # 学習しない層をフリーズ
    for layer in base_model.layers:
        if layer.name == 'conv_pw_5':
            break # この層以降はフリーズしない
        if isinstance(layer, tf.keras.layers.BatchNormalization):
             continue  # BNは常にフリーズしない
        layer.trainable = False
    x = base_model.output

    # 平均プーリング演算
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 全結合
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    # モデル生成
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    return model

def _main(input_shape, num_classes, epochs, batch_size):
    """
    学習プログラム

    Parameters
    ----------
    input_shape: tuple
        入力画像の形状
    num_classes : int
        クラス数
    epochs : int
        エポック数
    batch_size : int
        バッチサイズ
    """

    idg_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,  
        # ↑学習と推論で同じスケールにする必要がある。
        # ↑転移学習の場合はベースモデルのスケールと合わせる必要がある。
    )
    idg_validation = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0
    )

    gen_train = idg_train.flow_from_directory(
        directory="train",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )
    gen_validation = idg_validation.flow_from_directory(
        directory="val",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )
    print(gen_train.class_indices)

    # 学習モデル作成
    model = create_model_mobilenet(num_classes, input_shape=input_shape)

    # 最適化アルゴリズム
    optimizer = tf.keras.optimizers.Adam()

    # モデルの設定
    model.compile(
        loss='categorical_crossentropy', # 損失関数の設定
        optimizer=optimizer, # 最適化法の指定
        metrics=['acc'])

    # モデル情報表示
    model.summary()

    # モデルの学習
    history = model.fit(
              gen_train,
              epochs=epochs,
              validation_data=gen_validation,
              validation_steps=1,
              batch_size=batch_size)

    # モデル保存
    # model_name = 'model_' + os.path.splitext(os.path.basename(__file__))[0] + '.h5'
    model_name = 'model.h5'
    model.save(model_name)

    # 学習結果グラフ出力
    plot_result(history)

    # 評価 & 評価結果出力
    score = model.evaluate(gen_validation)
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# エントリポイント
if __name__ == "__main__":
    # データ枚数
    # 入力画像の形状
    input_shape = (64, 64, 3)
    # 分類クラス数
    num_classes = 15
    # エポック数
    epochs = 10
    # バッチサイズ
    batch_size = 16
    _main(input_shape, num_classes, epochs, batch_size)
