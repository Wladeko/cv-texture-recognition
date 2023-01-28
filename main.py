from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import accuracy_score

# from skimage.feature import daisy, hog, sift, surf
import numpy as np
import os
import cv2
from tqdm import tqdm

class_dict = {
    "aluminium_foil": 1,
    "brown_bread": 2,
    "corduroy": 3,
    "cork": 4,
    "cotton": 5,
    "cracker": 6,
    "linen": 7,
    "orange_peel": 8,
    "sponge": 9,
    "styrofoam": 10,
    "wool": 11,
}


def load_features(mode):
    images = []
    labels = []
    training_path = ".\\data\\train" if mode == "train" else ".\\data\\valid"

    for label in os.listdir(training_path):
        label_training_path = training_path + f"\\{label}"
        for image in os.listdir(label_training_path):
            image_label_training_path = label_training_path + f"\\{image}"
            img = cv2.imread(image_label_training_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append((img, img_gray))
            labels.append(class_dict[label])

    # Extract features
    features = []
    for image_tuple in tqdm(images):
        image = image_tuple[1]
        # Statistical features
        mean = np.mean(image)
        std = np.std(image)
        skewness = skew(image.flatten())
        kurtosis_ = kurtosis(image.flatten())

        # GLCM features
        glcm = graycomatrix(
            image,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, np.pi / 4, np.pi / 2],
            256,
            symmetric=True,
            normed=True,
        )
        contrast = graycoprops(glcm, "contrast").flatten()
        correlation = graycoprops(glcm, "correlation").flatten()
        energy = graycoprops(glcm, "energy").flatten()
        homogeneity = graycoprops(glcm, "homogeneity").flatten()
        dissimilarity = graycoprops(glcm, "dissimilarity").flatten()
        asm = graycoprops(glcm, "ASM").flatten()

        # glcm = graycomatrix(
        #     image,
        #     [5],
        #     [0],
        #     256,
        #     symmetric=True,
        #     normed=True,
        # )
        # contrast = graycoprops(glcm, "contrast").flatten()
        # correlation = graycoprops(glcm, "correlation").flatten()
        # energy = graycoprops(glcm, "energy").flatten()
        # homogeneity = graycoprops(glcm, "homogeneity").flatten()
        # dissimilarity = graycoprops(glcm, "dissimilarity").flatten()
        # asm = graycoprops(glcm, "ASM").flatten()

        # LBP features
        # radius = 3
        # n_points = 8 * radius
        # lbp = local_binary_pattern(image, n_points, radius)
        # lbp_hist = np.histogram(
        #     lbp, density=True, bins=n_points + 2, range=(0, n_points + 2)
        # )[0]

        # Extract SIFT features
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp1, des1 = sift.detectAndCompute(image, None)

        # # Extract SURF features
        # surf = cv2.xfeatures2d.SURF_create()
        # kp2, des2 = surf.detectAndCompute(image, None)

        # # Extract ORB features
        # orb = cv2.ORB_create()
        # kp3, des3 = orb.detectAndCompute(image, None)

        # Scale features
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_features = scaler.fit_transform(
        #     np.concatenate(
        #         (
        #             mean,
        #             std,
        #             skewness,
        #             kurtosis_,
        #             contrast,
        #             correlation,
        #             energy,
        #             homogeneity,
        #             # lbp_hist,
        #             # des1.flatten(),
        #             # des2.flatten(),
        #             # des3.flatten(),
        #         )
        #     )
        # )

        features_ = [
            mean,
            std,
            skewness,
            kurtosis_,
        ]
        features_.extend(contrast)
        features_.extend(correlation)
        features_.extend(energy)
        features_.extend(homogeneity)
        features_.extend(dissimilarity)
        features_.extend(asm)
        # features_.extend(lbp_hist)

        features.append(features_)

    features = np.array(features)
    labels = np.array(labels)
    # print(features.shape)
    # print(labels.shape)

    return features, labels
    # Load images and labels


# Split the data into training and validation sets
X_train, y_train = load_features("train")
X_valid, y_valid = load_features("valid")


# Initialize the classifier
clf = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels for the validation data
y_pred = clf.predict(X_valid)

# Calculate the accuracy of the classifier on the validation data
acc = accuracy_score(y_valid, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))
# print(y_pred[:20])
# print(y_valid[:20])
