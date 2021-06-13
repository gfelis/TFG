from utils import *

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img, model):
    return model.layers[2](model.layers[1](model.layers[0](process(img)))).numpy()[0]

def get_class(prediction: np.ndarray):
    classes = norm_test.columns[1:]
    percentage = prediction.max()
    class_index = np.where(prediction == percentage)[0][0]
    return percentage, classes[class_index], prediction

def random_sample_test(model, sample_len=10):
    test_images_paths = [TRAIN_IMAGES_FOLDER + path for path in random.sample(list(norm_test['image'].values), sample_len)]
    test_images = [read_image(path) for path in test_images_paths]
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        for image, index in zip(test_images, range(len(test_images))):  
            prediction = predict(image, model)
            accuracy, label_predicted, rest = get_class(prediction) #rest of probabilities of classes in rest
            print(accuracy, label_predicted)
            image_id = test_images_paths[index].split("/")[3]
            print(norm_test.loc[norm_test['image'] == image_id])
    

if __name__ == "__main__":
    data, train, test = load_split_dataset()
    norm_train = normalise_from_dataset_disjoint(train)
    norm_test = normalise_from_dataset_disjoint(test)

    #Aquest model confon apple_cider_rust amb complex, els intercanvia amb 99% d'accuracy
    dense_net = load_model("dense_net.h5")

    random_sample_test(dense_net)



 

