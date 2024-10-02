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
    モデル作成(MobileNet)
    
    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : tuple
        入力画像サイズ
    """
    # ImageNetで事前学習したMobileNetモデルを読込
    base_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet',
                                                           include_top=False,
                                                           input_shape=input_shape)
    # 一部の層をフリーズ解除（微調整のため）
    for layer in base_model.layers:
        layer.trainable = True  # すべての層を学習可能に設定

    x = base_model.output

    # 平均プーリング演算
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # ドロップアウトを追加（過学習防止）
    x = tf.keras.layers.Dropout(0.5)(x)
    # 全結合
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    # 出力層
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

    # データ拡張を含むImageDataGenerator
    idg_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # モデルの設定
    model.compile(
        loss='categorical_crossentropy',  # 損失関数の設定
        optimizer=optimizer,              # 最適化法の指定
        metrics=['acc'])

    # コールバックの設定
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    # モデル情報表示
    model.summary()

    # モデルの学習
    history = model.fit(
        gen_train,
        epochs=epochs,
        validation_data=gen_validation,
        callbacks=callbacks,
        batch_size=batch_size)

    # モデル保存
    model_name = 'model.keras'
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
    # 入力画像の形状
    input_shape = (128, 128, 3)
    # 分類クラス数
    num_classes = 15
    # エポック数
    epochs = 30
    # バッチサイズ
    batch_size = 16
    _main(input_shape, num_classes, epochs, batch_size)
