import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_image(name):
    img = cv2.imread(name, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def show(image):
    cv2.imshow('', image)
    cv2.waitKey(0)


def save(path, image):
    cv2.imwrite(path, image)


def canny_edge_detection(image, min=90, max=90):
    edges = cv2.Canny(image, min, max)
    return edges


def dilate(image, kernel, iterations=1):
    return cv2.dilate(image, kernel, iterations)


def generate_kernel(width, height):
    return np.ones((height, width), np.uint8)


def plot(image1, image2):
    plt.subplot(121), plt.imshow(image1, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image2, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def contours(image):
    return cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


def draw_contours(image, cons, thickness=1):
    return cv2.drawContours(image, cons, -1, (0,255,0), thickness)


def main():
    image = load_image('resource/pod.png')
    edges = canny_edge_detection(image, 100, 100)
    # show(edges)
    dil_edges = dilate(edges, generate_kernel(3, 3), 2)

    ret, thresh = cv2.threshold(dil_edges, 127, 255, 0)
    cons, hir = contours(thresh)
    print(cons, hir)
    con_edges = draw_contours(thresh, cons, 1)
    show(con_edges)
    plot(edges, con_edges)
    # save('resource/edges_pod.png', edges)

    pass


if __name__ == '__main__':
    main()
    print('started')
    pass
