from face_landmark.face_landmark_detection import LandmarkDetector
from scipy.misc import imread
from neural.neural_style_tutorial import run_style_transfer, image_loader, image_loader_image
import torchvision.models as models
import torch
import torchvision.transforms as transforms

cnn = models.vgg19(pretrained=True).features
dtype = torch.FloatTensor


def _square_size(top, bottom, left, right):
    return bottom - top if bottom - top > right - left else right - left


# face_image will return as numpy.ndarray
def detect_face_image_face_borders_points_face_size(image_path):
    im_array = imread(image_path, mode='RGBA')
    d = LandmarkDetector(im_array[:, :, :3])
    item = []
    face = []
    size = 0
    if len(d.faces) is not 0:
        d = d.faces.pop()
        size = _square_size(d.get('top'), d.get('bottom'), d.get('left'), d.get('right'))
        item = im_array[d.get('top'):d.get('top') + size, d.get('left'):d.get('left') + size]
        face = [(point[0] - d.get('left'), point[1] - d.get('top')) for point in d.get('points_list')]

    return item, face, size


def style_transfer(content_img, style_img):
    input_img_torch = content_img.clone()
    return run_style_transfer(cnn, content_img, style_img, input_img_torch)


def numpy_ndarray_as_torch_variable(img_numpy_ndarray):
    return image_loader(img_numpy_ndarray[:, :, :3]).type(dtype)


def image_as_torch_variable(img_path, size):
    return image_loader_image(img_path, size).type(dtype)


def torch_variable_as_pil(img_variable, size):
    unloader = transforms.ToPILImage()
    newImage = img_variable.view(3, size, size)  # remove the fake batch dimension
    newImage = unloader(newImage)
    newImage = newImage.convert('RGBA')
    return newImage









