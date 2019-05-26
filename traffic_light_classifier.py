import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import helpers
#import test_functions


## Visualize and Check images
def get_image_by_color(img_list, color, skip_num=0):
    count = 0
    for img in img_list:
        if img[1] != color:
            continue
            
        if count < skip_num:
            count += 1
            continue
        else:
            image = img[0]
            break
    return image


## Standardize the input images
def standardize_input(image):
    # resize image to the desired input size: 32x32(px)
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    return standard_im

## Standardize the output
def one_hot_encode(label):
    if label == "red":
        return [1, 0, 0]
    elif label == "yellow":
        return [0, 1, 0]
    elif label == "green":
        return [0, 0, 1]
    else:
        raise ValueError("input is invalid.. you should choice from red, yellow or green.")

## Convert original image list to STANDARDIZED_LIST
def standardize(image_list):
    standard_list = []
    for item in image_list:
        standardized_img = standardize_input(item[0])
        standardized_label = one_hot_encode(item[1])
        standard_list.append((standardized_img, standardized_label))
    return standard_list

def convert_to_hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

## Show images related with convering RGB to HSV
def show_converted_result(rgb):
    
    hsv = convert_to_hsv(rgb)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25,10))
    ax1.set_title('Standardized RGB image')
    ax1.imshow(rgb)
    ax2.set_title('HSV image')
    ax2.imshow(hsv)
    ax3.set_title('H channel')
    ax3.imshow(h, cmap='gray')
    ax4.set_title('S channel')
    ax4.imshow(s, cmap='gray')
    ax5.set_title('V channel')
    ax5.imshow(v, cmap='gray')

## Create Brightness Mask
def get_brightness_mask(hsv):
    return cv2.inRange(hsv, (0, 31, 155), (180, 256, 256))

## Create Noise Mask
def get_noise_mask(hsv):
    m_frame = cv2.inRange(hsv, (101, 38, 154), (114, 91, 249))
    m_red1 = cv2.inRange(hsv, (0, 30, 155), (14, 55, 230))
    m_red2 = cv2.inRange(hsv, (158, 30, 155), (180, 55, 230))
    m_dark_red = cv2.inRange(hsv, (108, 31, 162), (133, 53, 220))
    m_blue = cv2.inRange(hsv, (101, 90, 150), (108, 112, 225))
    m_gunjyo = cv2.inRange(hsv, (101, 67, 181), (108, 97, 206))
    m_blue_gray = cv2.inRange(hsv, (96, 29, 155), (114, 59, 214))
    m_light_blue = cv2.inRange(hsv, (93, 30, 214), (110, 68, 255))
    
    return m_frame + m_red1 + m_red2 + m_dark_red + m_blue + m_gunjyo + m_blue_gray + m_light_blue

## Filter noises
def noise_filtering(rgb, mask):
    copy = np.copy(rgb)
    copy[mask!=0] = [0, 0, 0]
    return copy

## Calculate probabilities by brightness feature
def get_probabilities(mask):
    denominator = np.sum(mask)
    rows = np.sum(mask, axis=1)
    margin = 2
    
    p_red = np.sum(rows[margin:12]) / denominator
    p_yellow = np.sum(rows[12:21]) / denominator
    p_green = np.sum(rows[21:-margin]) / denominator
    
    return p_red, p_yellow, p_green

## Filtering the image
def get_masked_rgb(rgb, mask):
    masked_rgb = np.copy(rgb)
    masked_rgb[mask==0] = [0,0,0]
    return masked_rgb

## Predict by brightness feature
def estimate_label_by_brightness(rgb, hsv):
    bright = get_brightness_mask(hsv)
    filtered_rgb = get_masked_rgb(rgb, bright)
    p_red, p_yellow, p_green = get_probabilities(filtered_rgb)
    
    threshold = 0.54;
    if p_red > threshold:
        estimate = [1, 0, 0]
    elif p_yellow > threshold:
        estimate = [0, 1, 0]
    elif p_green > threshold:
        estimate = [0, 0, 1]
    else:
        estimate = None
    
    return estimate, p_red, p_yellow, p_green

## Predict by brightness feature
##  this method will be used when classifier couldn't predict by brightness feature.
def estimate_label_by_color(masked_rgb):
    h = convert_to_hsv(masked_rgb)[:,:,0]
    count = np.count_nonzero(h)
    avg = (np.sum(h)/count)
    if avg > 115:
        return [1,0,0]
    elif avg > 90:
        return [0,0,1]
    elif avg > 20:
        return [0,1,0]
    else:
        return [1,0,0]

def get_misclassified_images(test_images, console=False):
    
    misclassified_images_labels = []

    for idx, image in enumerate(test_images):
        img = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        hsv = convert_to_hsv(img)
        predicted_label, _,_,_ = estimate_label_by_brightness(img, hsv)
        if predicted_label is None:
            masked_rgb = get_masked_rgb(img, get_brightness_mask(hsv))
            masked_rgb = noise_filtering(masked_rgb, get_noise_mask(hsv))
            predicted_label = estimate_label_by_color(masked_rgb)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        if(predicted_label != true_label):
            misclassified_images_labels.append((img, predicted_label, true_label))
            if console:
                print("Dataset idx: {} <= Misclassify idx: {}".format(idx, len(misclassified_images_labels)-1))
            
    return misclassified_images_labels

## Image directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


## Load the training datasets
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

## Play with the input images and get senses how you extract features
image = get_image_by_color(IMAGE_LIST, "yellow")
if image is not None:
    plt.imshow(image)
    print(image.shape)

## Test ##
# tests = test_functions.Tests()
# tests.test_standardize_size(standardize_input, image)
# tests.test_one_hot(one_hot_encode)

## Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

## Convert a standardized data
index = 0
std_rgb = STANDARDIZED_LIST[index][0]
std_rgb_label = STANDARDIZED_LIST[index][1]

## Visualize converted standardized data
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.set_title("before: " + IMAGE_LIST[index][1])
ax1.imshow(IMAGE_LIST[index][0])
ax2.set_title("after: " + str(std_rgb_label))
ax2.imshow(std_rgb)


## Plot the original RGB image, HSV image and the three channels(h, s, v)
show_converted_result(std_rgb)
print('Label [red, yellow, green]: ' + str(std_rgb_label))


## Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)
random.shuffle(STANDARDIZED_TEST_LIST)

## Classify
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: {}'.format(str(accuracy)))
print("Number of misclassified images = {} out of {}.".format(str(len(MISCLASSIFIED)), str(total)))


## Test never classify any red lights as green
#tests = test_functions.Tests()
#if(len(MISCLASSIFIED) > 0):
#    tests.test_red_as_green(MISCLASSIFIED)
#else:
#    print("MISCLASSIFIED may not have been populated with images.")



## After this line of codes are for improve algorithm.
## See what lead the misclassify.
def print_misclassified_image(MISCLASSIFIED, idx):
    img, pred_label, true_label = MISCLASSIFIED[idx]
    plt.imshow(img)
    print("No{} => You predected {}, but the true label is {}.".format(idx, pred_label, true_label))


def print_misclassified_probabilities(misclassified_list):
    for idx, (img, pred_label, true_label) in enumerate(misclassified_list):
        _, pr, py, pg = estimate_label(img, convert_to_hsv(img))
        print("No{} => You predected {}, but the true label is {}.".format(idx, pred_label, true_label))
        print("Probabilities: red=> {} / yellow=> {} / green=> {}\n".format(pr, py, pg))
    
#print_misclassified_probabilities(MISCLASSIFIED)


