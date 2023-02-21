from glob import glob
import os
import pandas as pd

LOCAL_PATH = 'Data//test//10001//*'
IMAGE_PATH = 'Data//test//10001//{}//*.png'
IMAGES_PATH = 'Data/test/10001/{}/'
LABEL_PATH = 'Data//test//10001//{}//*.txt'



def get_image_info(path=LOCAL_PATH):
    image_id = []
    image_path = []
    labels_name = []
    labels_path = []
    for folder in glob(path):
        folder_name = os.path.basename(folder)
        # folder_path = os.path.join(IMAGES_PATH, folder_name)
        for image in glob(IMAGE_PATH.format(folder_name)):
            image_basename = os.path.basename(image)
            images_name = image_basename[:-4]
            images_path = os.path.join(IMAGES_PATH.format(folder_name), image_basename)
            labels = 1 if folder_name == 'live' else 0
            labels_name.append(labels)
            image_id.append(images_name)
            image_path.append(images_path)

        for bbox_file in glob(LABEL_PATH.format(folder_name)):
            bbox_file_name = os.path.basename(bbox_file)
            label_path = os.path.join(IMAGES_PATH.format(folder_name), bbox_file_name)
            labels_path.append(label_path)


    dict = {
        'image_id': image_id, 'image_path': image_path, 'label_path': labels_path, 'labels': labels_name
    }

    df = pd.DataFrame(dict)
    df.to_csv('data.csv', index=False)

get_image_info(LOCAL_PATH)