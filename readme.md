# This tool provides a simple method for detecting faces within an image and saving each individual face as a `512 x 512 square image`. This facilitates stable diffusion training. The tool utilizes the [deepface](https://github.com/serengil/deepface) model for face detection.

Sample jupyter notebook : `extract_convert_module.ipynb`

Command:
```bash
python extract_face_sq.py --input_folder='./input_img'
```

Extracted individual face images will be saved in the subfolder `resized` of the input image folder