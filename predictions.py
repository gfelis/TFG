from random import sample
from utils import *
from visuals import save_test_results, save_training_curves

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

def find_maximums(list):
    max1, max2, max3= 0, 0, 0	  
    for i in list:	
        if i > max1:	
            max3 = max2 
            max2 = max1 
            max1 = i 
        elif i > max2:	
            max3 = max2 
            max2 = i 
        elif i > max3:	
            max3 = i 
    return max2, max3

def get_class(prediction: np.ndarray):
    classes = pd.Index(['frog_eye_leaf_spot', 'healthy', 'scab', 'complex', 'powdery_mildew', 'rust'])
    percentage = prediction.max()
    class_index = np.where(prediction == percentage)[0][0]
    if percentage <= 0.75:
        second_percentage, third_percentage = find_maximums(prediction)
        if percentage - second_percentage <= 0.25:
            second_index = np.where(prediction == second_percentage)[0][0]
            if second_percentage - third_percentage <= 0.15:
                third_index = np.where(prediction == third_percentage)[0][0]
                return (percentage, second_percentage, third_percentage), classes[class_index] + ' ' + classes[second_index] + ' ' + classes[third_index], prediction
            return (percentage, second_percentage), classes[class_index] + ' ' + classes[second_index], prediction
    return [percentage], classes[class_index], prediction
            
def random_sample_test_joint(model, output_file, sample_len=10, seed=10):
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

    save_test_results(output_file, (correctly_predicted, one_label_correctly, two_labels_correctly, three_labels_correctly),
                                    (incorrectly_predicted, one_label_incorrectly, two_labels_incorrectly, three_labels_incorrectly))

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

    dense_net = load_model("dense_net_joint_2daug_dedup.h5")

    log = read_log('dense_net_joint_2daug_dedup.log')
    save_training_curves(log['categorical_accuracy'], log['val_categorical_accuracy'], 'DenseNet sense duplicació' , 20)

    
    statistics = random_sample_test_joint(dense_net, "single_test", 1, 10)
    statistics2 = random_sample_test_joint(dense_net, "ten_test", 10, 20)


    #print(f'Total Hits: {statistics[0]} Total Miss: {statistics[3]}')
    #print(f'One Hits: {statistics[1]} One Miss: {statistics[4]}')
    #print(f'Two Hits: {statistics[2]} Two Miss: {statistics[5]}')
    #print(f'Three Hits: {statistics[6]} Three Miss: {statistics[7]}')



 

