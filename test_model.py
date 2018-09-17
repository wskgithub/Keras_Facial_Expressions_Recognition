import numpy as np
from keras.models import load_model
from load_data import load_images

def classify_image(model, image_lists, kind_lists):
    """
    对测试集图片进行分类，并输出分类日志
    :param model: 读取进来的模型
    :param image_lists:需做分类的图片列表
    :param kind_lists:类型名称列表
    :return:null
    """
    # 从图片列表中遍历每一张需要识别的图片
    image_count = len(image_lists)
    true_count = 0
    for image in image_lists:
        result = model.predict(image[0])[0] # 将图片送入模型中预测
        proba = np.max(result) # 取出相似度最高的一项
        label = kind_lists[int(np.where(result == proba)[0])] # 获得识别出类型的标签
        # 打印分类log
        log = ("result:" + label + " -> " + str(proba * 100) 
            + " -> source:" + image[1] 
            + " -> name:" + image[2] 
            + " -> path:" + image[3]
            + "\n")
        print(log)
        # 判断识别结果是否正确
        if label == image[1]:
            true_count += 1
        else:
            # 输出分类错误日志到文件
            with open("logs/log.txt", "a") as f:
                f.write(log)

    print("分类图片总数:{}, 分类正确:{}, 分类正确率:{}%"
        .format(image_count, true_count, (true_count/image_count) * 100))

if __name__=='__main__':
    # 测试模型并输出分类日志
    test_data_path = "data/CK+48_test" # 测试集
    model_path = "models/ck+48_model.h5" # 测试模型
    image_lists = load_images(test_data_path, IMAGE_SIZE) # 加载测试图片
    model = load_model(model_path) # 加载测试模型
    classify_image(model, image_lists, KIND_LISTS)