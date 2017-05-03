import os

# using ImageMagick
#
# convert -resize x480 before.jpg after.jpg

output_size = '32x32'
image_dir = './images'

files = os.listdir(image_dir)
for file_name in files:
    image_path = "{}/{}".format(image_dir, file_name)
    cmd = "convert -resize {} {} {}".format(output_size, image_path, image_path)
    os.system(cmd)
    print(file_name)
