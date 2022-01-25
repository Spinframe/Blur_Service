from PIL import Image
from PIL import ImageDraw
from google.cloud import vision
import io
import cv2
from os import listdir
from os.path import isfile, join
import sys

def detect_faces(path , outpath):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    # read the image for the bluring
    img = cv2.imread(path)
    # get width and height of the image
    h, w = img.shape[:2]
    # gaussian blur kernel size depends on width and height of original image
    kernel_width = (w // 55) | 1
    kernel_height = (h // 55) | 1

    # Open an image for the mask and pasted the blure mask and the image
    im = Image.open(path)
    # Create rounded rectangle mask
    mask = Image.new('L', im.size, 0)
    #the Draw allow as to drow on the mask the ellipse of the face
    draw = ImageDraw.Draw(mask)

    for face in faces:
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])
        #print('face bounds: {}'.format(','.join(vertices)))

        start_x = int(vertices[0].split(',')[0].split('(')[1])
        start_y = int(vertices[0].split(',')[1].split(')')[0])
        end_x = int(vertices[2].split(',')[0].split('(')[1])
        end_y = int(vertices[2].split(',')[1].split(')')[0])
        draw.ellipse((start_x, start_y, end_x, end_y), fill=(255))

    #mask.save(maskpath)

    blur = cv2.GaussianBlur(img, (kernel_width, kernel_height), 0)
    #convert the cv2 blur to PIL format for pasted them together
    #im_blur = Image.fromarray(blur)
    cv2.imwrite('.\\blur.png', blur)
    im_blur = Image.open('.\\blur.png')
    im.paste(im_blur, mask=mask)
    im.save(outpath)


    if response.error.message:
        print(response.error.message)
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

if __name__ == '__main__':
    
    src = sys.argv[1]
    out = sys.argv[2]
    #print(src,out)
    '''onlyfiles = [f for f in listdir(src) if isfile(join(src, f))]
    for file in onlyfiles:
        src_file = src+file
        out_file = out + file
        out_mask = out+'mask_'+file
        detect_faces(src_file, out_file, out_mask)
        print(file+" secsses")
	'''
    detect_faces(src, out)
    print("ok")