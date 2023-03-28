from deepface import DeepFace

# import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse


def read_img_path(input_folder):
    """
    Read only images from input_folder and return a list of image paths,
    Take in jpg, JPG, jpeg, JPEG, png, PNG, bmp, BMP
    Exclude subfolders

    Args:
        input_folder (str): input image folder
    """
    img_paths = glob.glob(input_folder + "/*")
    img_paths = [path for path in img_paths if os.path.isfile(path)]
    img_paths = [
        path
        for path in img_paths
        if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    return img_paths


def split_filename(path):
    """
    Split the filename from the path
    return path and filename and extension

    Args:
        path (str): input image path
    """
    import os

    pathname, filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    return pathname, filename, ext


def convert_sq(x, y, w, h, scale=1):
    """Convert the detected rectangle to a square and
    return the coordinates of the square.
    Scale allows flexibility to expand/decrease the range of detected square region.

    Args:
        x (_type_): x-coordinate of the top-left corner
        y (_type_): y-coordinate of the top-left corner
        w (_type_): width
        h (_type_): height
        scale (int, optional): Expand/decrease the range of detected square region. Defaults to 1.

    Returns:
        x1, y1, x2, y2: return the coordinates of the square
    """

    cx = x + w // 2
    cy = y + h // 2
    cr = max(w, h) // 2

    r = cr * scale
    # cv2.rectangle(canvas, (cx-r, cy-r), (cx+r, cy+r), (0,255,0), 1)
    # croped = img[cy-r:cy+r, cx-r:cx+r]
    # cv2.imshow("croped{}".format(i), croped)

    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r

    return x1, y1, x2, y2


def resize_save_image(img_obj, img_path):
    """Resize and save image

    Args:
        img_obj (deepface obj): deepface object
        img_path (str): input image path
    """
    for i in range(len(img_obj)):
        # skip if error detected during cropping and resizing
        try:
            x = img_obj[i]["facial_area"]["x"]
            y = img_obj[i]["facial_area"]["y"]
            w = img_obj[i]["facial_area"]["w"]
            h = img_obj[i]["facial_area"]["h"]

            x1, y1, x2, y2 = convert_sq(x, y, w, h, 2)

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image1 = cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 5)
            crop_img = image[y1:y2, x1:x2]

            # Resize image
            img_resized = cv2.resize(crop_img, (512, 512))
            # Save image
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            pathname, filename, ext = split_filename(img_path)

            # create folder if it did not exist
            resized_folder = os.path.join(pathname, "resized")
            if not os.path.exists(resized_folder):
                os.makedirs(resized_folder)

            resized_image_file = os.path.join(
                resized_folder, filename + "_resized" + "_" + str(i) + ext
            )
            print(resized_image_file)

            cv2.imwrite(resized_image_file, img_resized)
        except:
            print("Error processing this image:", img_path + "_" + str(i))
            return


def extract_face(path):
    """Extract face from image and save it as a new square image in resized folder
    Args:
        path (str): input image path
    """
    img_path = path
    backends = [
        "opencv",
        "ssd",
        "dlib",
        "mtcnn",
        "retinaface",
        #   'mediapipe'
    ]

    # Skip if no face detected
    try:
        img_obj = DeepFace.extract_faces(
            img_path=img_path,
            target_size=(224, 224),
            detector_backend="retinaface",
            align=False,
        )
    except:
        print("No face detected for image:", img_path)
        return

    resize_save_image(img_obj, img_path)


def run(input_folder):
    """Run the image extraction in batches.

    Args:
        input_folder (str): input image folder
    """
    img_paths = read_img_path(input_folder)
    print(img_paths)
    for img_path in img_paths:
        extract_face(img_path)


if __name__ == "__main__":
    """Example to run the script:
    python extract_face_sq.py --input_folder './input_img'
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./",
        help="Input directory of images",
    )
    args = parser.parse_args()
    print(args)
    run(args.input_folder)
