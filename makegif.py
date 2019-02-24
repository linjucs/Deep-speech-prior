import imageio
from PIL import Image
import numpy as np

input_folder = ''
output_name = 'speech'
output_size = 500

def make_gif(output_file=output_name+'.gif', folder=input_folder+'output/', output_size=output_size):
    imgs = []
    for i in range(0,1000, 10):
        img = Image.open('{}pre_{}.jpg'.format(folder,i))
        img = img.resize((output_size, output_size))
        img = np.array(img)
        imgs.append(img)
    imageio.mimsave(output_file, imgs, duration=0.5)

def img_parse(input_file, output_file, output_size=output_size):
    img = Image.open(input_file)
    img = img.resize((output_size, output_size))
    img.save(output_file)


if __name__=='__main__':
    make_gif()
