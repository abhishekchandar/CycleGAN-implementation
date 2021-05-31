#code for showing result
import util
import config
from generator_model import Generator
import torch
from torch.autograd import Variable
import PIL
from PIL import Image
import numpy as np


PRETRAINED_MODEL = "genm.pth.tar"

TEST_IMAGE_DIR = '../dataset/train/monet/0a5075d42a.jpg'


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    dict_items = checkpoint.items()
    # test = list(dict_items)[:1]
    # print(test)
    model = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def tensor_to_image(tensor):
    tensor = tensor.cpu()
    tensor = tensor*255
    if np.ndim(tensor)>3:
        tensor = torch.squeeze(tensor)
        print(tensor.shape)
        tensor = np.array(tensor, dtype=np.uint8)
        print(tensor.shape)
    return PIL.Image.fromarray(tensor.T,'RGB')

if config.LOAD_MODEL:
    image = np.array(Image.open(TEST_IMAGE_DIR).convert('RGB'))
    transformed = config.transforms_single(image=image)
    test_input = Variable(torch.Tensor(transformed['image']).unsqueeze(0)).to(config.DEVICE)
    model = load_checkpoint(PRETRAINED_MODEL)
    output = model(test_input)
    tensor_to_image(output).show()
    tensor_to_image(output).save('output.png')