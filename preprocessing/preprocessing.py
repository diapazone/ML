from PIL import Image, ImageOps
import os
import cv2 as cv
import numpy as np
import pywt
from matplotlib import pyplot as plt

def init():
    # Путь к папке с исходными снимками
    first_input_folder = ''
    second_input_folder = ''
    # Путь к папке для сохранения обработанных снимков
    output_folder = ''
    return first_input_folder, second_input_folder, output_folder

def crop(folder = 'тест'):
    # Проходим по всем файлам в папке с исходными снимками
    for filename in os.listdir(folder):
        image = Image.open(os.path.join(folder, filename))
        # Получаем ширину и высоту изображения
        width, height = image.size
        # Вычисляем координаты для обрезки изображения по центру
        left = (width - 3000) // 2
        top = (height - 3000) // 2
        right = (width + 3000) // 2
        bottom = (height + 3000) // 2
        # Обрезаем изображение
        image = image.crop((left, top, right, bottom))
        image.save(os.path.join(folder, filename))

def bmp_to_jpeg(input_folder, output_folder, new_name):
    count = 0
    # Проходим по всем файлам в папке с исходными снимками
    for filename in os.listdir(input_folder):
        # Открываем изображение и сохраняем в формате JPEG
        if filename.endswith('.bmp') or filename.endswith('.tif'):
            count+=1
            image = Image.open(os.path.join(input_folder, filename))
            image = image.convert('L')
            new_filename = new_name + str(count) + '.jpg'
            image.save(os.path.join(output_folder, new_filename), 'JPEG', quality=100)

def tif_to_jpeg(input_folder, output_folder, new_name):
    count = 0
    # Проходим по всем файлам в папке с исходными снимками
    for filename in os.listdir(input_folder):
        # Открываем изображение и сохраняем в формате JPEG
        if filename.endswith('.bmp') or filename.endswith('.tif'):
            count+=1
            image = Image.open(os.path.join(input_folder, filename))
            new_filename = new_name + str(count) + '.jpg'
            image.mode = 'I'
            image.point(lambda i:i*(1./256)).convert('L')\
                               .save(os.path.join(output_folder, new_filename), 'JPEG', quality=100)

def bmp_to_png(input_folder, output_folder, new_name):
    count = 0
    # Проходим по всем файлам в папке с исходными снимками
    for filename in os.listdir(input_folder):
        # Открываем изображение и сохраняем в формате PNG
        if filename.endswith('.bmp') or filename.endswith('.tif'):
            count+=1
            image = Image.open(os.path.join(input_folder, filename))
            image = image.convert('L')
            new_filename = new_name + str(count) + '.png'
            image.save(os.path.join(output_folder, new_filename), 'PNG')

def tif_to_png(input_folder, output_folder, new_name):
    count = 0
    # Проходим по всем файлам в папке с исходными снимками
    for filename in os.listdir(input_folder):
        # Открываем изображение и сохраняем в формате PNG
        if filename.endswith('.bmp') or filename.endswith('.tif'):
            count+=1
            image = Image.open(os.path.join(input_folder, filename))
            new_filename = new_name + str(count) + '.png'
            image.mode = 'I'
            image.point(lambda i:i*(1./256)).convert('L')\
                               .save(os.path.join(output_folder, new_filename), 'PNG')

def test():
    img = cv.imread('bad_1.png', cv.IMREAD_GRAYSCALE)
    # global thresholding
    ret1,th1 = cv.threshold(img,64,255,cv.THRESH_BINARY)
    cv.imwrite('bin64.png', th1)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('otsu.png', th2)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('gaus_otsu.png', th3)
    # plot all the images and their histograms
    images = [img, 0, th1,
                 img, 0, th2,
                 blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
                 'Original Noisy Image','Histogram',"Otsu's Thresholding",
                 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        plt.show()
    # Otsu's thresholding after median filtering
    blur = cv.medianBlur(img, 3)
    ret4,th4 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('median_otsu.png', th4)
    coeffs = pywt.dwt2(img, 'haar')
    coeffs = list(coeffs)
    coeffs[0] = np.zeros_like(coeffs[0])
    wavelet = pywt.idwt2(coeffs, 'haar')
    wavelet = cv.convertScaleAbs(wavelet)
    ret5,th5 = cv.threshold(wavelet,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('wavelet_otsu.png', th5)
    br_correct = cv.convertScaleAbs(img, alpha=1.0, beta=0)
    ret6,th6 = cv.threshold(br_correct,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imwrite('br_correct_otsu.png', th6)


def main():
    # инициализируем пути
    first_input_folder, second_input_folder, output_folder = init()
    # переводим все фотографии в формат jpeg или png
    # с помощью соответствующих функций
    test()
    crop()
    return 0

main()