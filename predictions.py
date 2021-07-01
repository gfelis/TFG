from utils import *
from visuals import save_test_results

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)

def predict(img, model):
    return model.layers[2](model.layers[1](model.layers[0](process(img)))).numpy()[0]

def find_second_max(list):
    count = 0
    m1 = m2 = float('-inf')
    for x in list:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

def get_class(prediction: np.ndarray):
    classes = pd.Index(['rust', 'powdery_mildew', 'frog_eye_leaf_spot', 'complex', 'scab', 'healthy'])
    percentage = prediction.max()
    class_index = np.where(prediction == percentage)[0][0]
    if percentage <= 0.55:
        second_percentage = find_second_max(prediction)
        second_index = np.where(prediction == second_percentage)[0][0]
        return (percentage, second_percentage), classes[class_index] + ' ' + classes[second_index], prediction
    return [percentage], classes[class_index], prediction
            
def random_sample_test_joint(model, sample_len=10, seed=10):
    random.seed(seed)
    test_images_paths = [TRAIN_IMAGES_FOLDER + path for path in random.sample(list(norm_test['image'].values), sample_len)]
    
    correctly_predicted = 0
    incorrectly_predicted = 0
    one_label_correctly = 0
    one_label_incorrectly = 0
    two_labels_correctly = 0
    two_labels_incorrectly = 0
    three_labels_correctly = 0
    three_labels_incorrectly = 0

    for index, path in enumerate(test_images_paths):  
            image = read_image(path)
            prediction = predict(image, model)
            accuracy, label_predicted, rest = get_class(prediction) #rest of probabilities of classes in rest
            image_id = test_images_paths[index].split("/")[3]
            expected_classes = data.loc[data['image'] == image_id]['labels'].values[0]

            labels = set(label_predicted.split())
            correct_labels = set(expected_classes.split())

            if labels == correct_labels:
                correctly_predicted+=1
                if len(correct_labels) == 3: three_labels_correctly+=1
                if len(correct_labels) == 2: two_labels_correctly+=1
                if len(correct_labels) == 1: one_label_correctly+=1
            else:
                incorrectly_predicted+=1
                if len(correct_labels) == 3: three_labels_incorrectly+=1
                if len(correct_labels) == 2: two_labels_incorrectly+=1
                if len(correct_labels) == 1: one_label_incorrectly+=1
                

            print('====================')
            print(f'Predicted: {label_predicted} with {accuracy}, other predictions: {rest}')
            print(f'Correct labels are: {expected_classes} ')

    return (correctly_predicted, one_label_correctly, two_labels_correctly, incorrectly_predicted, 
    one_label_incorrectly, two_labels_incorrectly, three_labels_correctly, three_labels_incorrectly)

def full_test_joint(model, output_file):
    test_images_paths = [TRAIN_IMAGES_FOLDER + img for img in list(norm_test['image'].values)]
    
    correctly_predicted = 0
    incorrectly_predicted = 0
    one_label_correctly = 0
    one_label_incorrectly = 0
    two_labels_correctly = 0
    two_labels_incorrectly = 0
    three_labels_correctly = 0
    three_labels_incorrectly = 0
    

    for index, path in enumerate(test_images_paths):
            image = read_image(path)
            print("Processing: " + str(index) + "/" + str(len(test_images_paths)))
            prediction = predict(image, model)
            accuracy, label_predicted, rest = get_class(prediction) #rest of probabilities of classes in rest
            image_id = test_images_paths[index].split("/")[3]
            expected_classes = data.loc[data['image'] == image_id]['labels'].values[0]

            labels = set(label_predicted.split())
            correct_labels = set(expected_classes.split())

            if labels == correct_labels:
                correctly_predicted+=1
                if len(correct_labels) == 3: three_labels_correctly+=1
                if len(correct_labels) == 2: two_labels_correctly+=1
                if len(correct_labels) == 1: one_label_correctly+=1
            else:
                incorrectly_predicted+=1
                if len(correct_labels) == 3: three_labels_incorrectly+=1
                if len(correct_labels) == 2: two_labels_incorrectly+=1
                if len(correct_labels) == 1: one_label_incorrectly+=1        

    save_test_results(output_file, (correctly_predicted, one_label_correctly, two_labels_correctly, three_labels_correctly),
                                    (incorrectly_predicted, one_label_incorrectly, two_labels_incorrectly, three_labels_incorrectly))

    return (correctly_predicted, one_label_correctly, two_labels_correctly, incorrectly_predicted, 
    one_label_incorrectly, two_labels_incorrectly, three_labels_correctly, three_labels_incorrectly)

if __name__ == "__main__":
    data, train, test = load_split_dataset()
    norm_train = normalise_from_dataset_joint(train)
    norm_test = normalise_from_dataset_joint(test)

    dense_net = load_model("dense_net_joint_2daug.h5")

    history_dense_net = read_log('efn_joint_2daug.log')

    
    statistics = full_test_joint(dense_net)


    print(f'Total Hits: {statistics[0]} Total Miss: {statistics[3]}')
    print(f'One Hits: {statistics[1]} One Miss: {statistics[4]}')
    print(f'Two Hits: {statistics[2]} Two Miss: {statistics[5]}')
    print(f'Three Hits: {statistics[6]} Three Miss: {statistics[7]}')



 

