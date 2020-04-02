
import shutil
from skimage import io
import cv2
import os
import matplotlib.pyplot as plt

def distances_ssim(directory, target, si, distance_flag):
    try:
        names = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory, i))]
        num = len(names)
        distances = []
        im1 = cv2.imread(target, 0)
        dims = im1.shape
        orb = cv2.ORB_create(200, 1.2)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
        kp1, des1 = orb.detectAndCompute(im1, None)
        for i in range(0, num):
            im2 = cv2.imread(directory + names[i], 0)
            if im2.shape != dims:
                im2 = cv2.resize(im2, dims)
            kp2, des2 = orb.detectAndCompute(im2, None)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            dist = [y.distance for y in matches[:50]]
            distances.append([names[i], sum(dist)])
        distances.sort(key=lambda distances: distances[1])
        return distances[:si]
    except Exception as e:
        print(e)

def move_sample(distances, directory, new_name):
    try:
        if not os.path.isfile(new_name):
            os.mkdir(new_name)
        files = []
        num = 0
        for i, _ in distances:
            if os.path.isfile(directory + i):
                shutil.copy(directory + i, new_name + i)
                # python train.py --dataroot ./datasets/techno2jazz --name testTech2Jazz --model cycle_gan --gpu_ids -1
                if num < 100:
                    files.append(directory + i)
                    num += 1
        return files
    except Exception as e:
        print(e)

def plot_cluster(files, label, method, nrows=10, ncols=10):
    plt.figure(figsize=(ncols, nrows))
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(io.imread(files[i]))
        plt.axis('off')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    try:
        genre = 'metal'
        target = './metal/albums/B0000279YD.jpg'
        dists = distances_ssim('./' + genre + '/albums/', target, 500, False)

        move_sample(dists[:400], './metal/albums/', './metal/feature_train/')
        move_sample(dists[400:], './metal/albums/', './metal/feature_test/')

    except Exception as e:
        print(e)