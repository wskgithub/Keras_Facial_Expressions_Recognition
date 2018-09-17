from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def initialize_model(filters, kernel_size, input_shape, pool_size, nb_classes):
    """
    初始化模型，构建卷积层和全连接层
    :param filters:卷积滤波器数量
    :param kernel_size:卷积核大小
    :param input_shape:图像张量
    :param pool_size:池化缩小比例因素
    :param nb_classes:分类数
    :return:初始化后的CNN模型
    """
    print("[INFO] initialize model...")
    # 生成模型
    model = Sequential()

    #####特征层#####
    # 卷积层
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu'))
    # 池化
    model.add(MaxPooling2D(pool_size=pool_size))
    # 卷积层
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu'))
    model.add(Conv2D(filters=filters*2, kernel_size=kernel_size, activation='relu'))
    # 池化
    model.add(MaxPooling2D(pool_size=pool_size))
    # 卷积层
    model.add(Conv2D(filters=filters*4, kernel_size=kernel_size, activation='relu'))
    model.add(Conv2D(filters=filters*4, kernel_size=kernel_size, activation='relu'))
    #池化
    model.add(MaxPooling2D(pool_size=pool_size))
    #####全链接层#####
    # 压缩维度
    model.add(Flatten())
    # 全链接层
    model.add(Dense(128, activation='relu'))
    # 模型平均，防止过拟合
    model.add(Dropout(0.5))
    # Softmax分类
    model.add(Dense(nb_classes, activation='softmax'))
    # 输出模型结构
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


def train(model, x_train, y_train, x_test, y_test, batch_size, epochs, model_name):
    """
    训练模型
    对训练的图像做图像增强
    :param model:需训练的模型
    :param x_train:训练数据
    :param y_train:训练数据标签
    :param x_test:验证数据
    :param y_test:验证数据标签
    :param batch_size:一批数据的大小
    :param epochs:循环的次数
    :param model_name:模型名称
    :return:null
    """
    # 使用TensorBoard对训练过程进行可视化
    tb = TensorBoard(log_dir='./logs/tensorboard',  # log 目录
                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)
    # 提前结束训练
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    callbacks = [tb]
    
    # 配置训练模型
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'], loss='categorical_crossentropy')
    # 图像扩充，左右随机裁剪20%像素，以30度的角度进行旋转
    data_gen = ImageDataGenerator(rotation_range = 30, width_shift_range=0.2, height_shift_range=0.2)
    # 逐批生成数据训练模型
    model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                       steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                       verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
    # 训练结束保存模型
    print("[INFO] save model...")
    model.save(model_name)