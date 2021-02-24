import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
cv2.ocl.setUseOpenCL(False)
from PIL import Image, ImageEnhance
from sklearn.cluster import MiniBatchKMeans
import os
import base64

from pathlib import Path
from utility import(
    img_to_bytes,
    read_markdown_file,
)

#function to load image
@st.cache
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def load_image(img):
    im =Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('facedata/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('facedata/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('facedata/haarcascade_smile.xml')


#define function for canny edge detection
def canny_edge(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)

    return edges


#function to detect faces
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #draw rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    return img, faces

#function to detect eyes
def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img


#function to detect smile


def detect_smiles(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = img[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
        return img



#quant function
def color_quantization(our_image):
    new_img = np.array(our_image.convert('RGB'))
    image = cv2.cvtColor(new_img,1)
    (h, w) = image.shape[:2]
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = 4)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return quant



#main function where all our streamlit code will run
def main():
    """Image Operations UI Application"""

    activities = ["Home","Filters", "Image Processing", "Detect Face Components", "Color Quantization", "About Me"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.write("")
        intro_markdown = read_markdown_file("markdowns/welcome.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)
            

    elif choice == "Filters":
        html_temp = """<div style= "background-color: #0a366b;padding:3px 10px; text-align: center; margin-bottom: 15px">
        <h2 style= "color: white; font-weight: 700">Simple Image Enhancement Filters</h2>"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write(" ")

        image_file = st.file_uploader("Upload an Image", type = ['jpg', 'jpeg', 'png'])
        if image_file is not None:
            filter_image = Image.open(image_file)
            enhance_type = st.sidebar.radio("Enhance Your Image", ["Original", "Gray-Scale", "Contrast", "Brightness", "Sharpness", "Blurring"])
            if enhance_type == "Original":
                st.text("Your Original Image")
                st.write(type(filter_image))
                st.image(filter_image)

        #enhance_type = st.sidebar.radio("Enhance Your Image", ["Original", "Gray-Scale", "Contrast", "Brightness", "Sharpness", "Blurring"])
            if enhance_type == "Gray-Scale":
                gray_img = np.array(filter_image.convert('RGB'))
                new_img = cv2.cvtColor(gray_img, 1)
                gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                #st.write(gray_img)
                st.write("Your Gray Scale Image Below: ")
                st.image(gray)

            if enhance_type == "Contrast":
                #take a silder to have contrast rate
                contrast_rate = st.sidebar.slider("Adjust Contrast", 0.5, 3.5)
                contrast_enhancer = ImageEnhance.Contrast(filter_image)
                contrast_image = contrast_enhancer.enhance(contrast_rate)
                st.write("Your Contrast Image is Below: ")
                st.image(contrast_image)

            if enhance_type == "Brightness":
                #take a slider to have brightness rate
                brightness_rate = st.sidebar.slider("Adjust Brightness", 0.5, 3.5)
                brightness_enhancer = ImageEnhance.Brightness(filter_image)
                brightness_image = brightness_enhancer.enhance(brightness_rate)
                st.write("Your Brighter Image is Below: ")
                st.image(brightness_image)

            if enhance_type == "Sharpness":
                sharpness_rate = st.sidebar.slider("Adjust Sharpness", 0.5, 3.5)
                sharpness_enhancer = ImageEnhance.Sharpness(filter_image)
                sharpness_image = sharpness_enhancer.enhance(sharpness_rate)
                st.write("Your Sharper Image is Below: ")
                st.image(sharpness_image)

            if enhance_type == "Blurring":
                reimg = np.array(filter_image.convert('RGB'))
                blur_rate = st.sidebar.slider("Adjust Blur Rate", 0.5, 3.5)
                store_img = cv2.cvtColor(reimg, 1)
                blur_img = cv2.GaussianBlur(store_img, (11,11), blur_rate)
                st.write('Blurred Image is Below: ')
                st.image(blur_img)



    elif choice == "Image Processing":
        html_temp = """<div style= "background-color: #0a366b;padding:3px 10px; text-align: center; margin-bottom: 15px">
        <h2 style= "color: white; font-weight: 700">Perform Image Processing Tasks</h2>"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write(" ")
        #upload a new image file for preprocessing
        processing_image_file = st.file_uploader("Upload Any Image", type = ['jpg', 'jpeg', 'png'])
        if processing_image_file is not None:
            process_image = Image.open(processing_image_file)
            processing_tasks = st.sidebar.radio("Select Processing Tasks", ["Real", "Canny Edge Detection", "Image Pyramids"])
            if processing_tasks == "Real":
                st.text("Your Image")
                st.image(process_image)


            elif processing_tasks == "Canny Edge Detection":
                canny_edge_result = canny_edge(process_image)
                st.image(canny_edge_result)

            elif processing_tasks == "Image Pyramids":
                new_pyramid_image = np.array(process_image.convert('RGB'))
                img_pyramid = cv2.cvtColor(new_pyramid_image, 1)
                layer = img_pyramid.copy()
                for i in range(4): 
                    #plt.subplot(2, 2, i + 1) 
                    # using pyrDown() function 
                    layer = cv2.pyrDown(layer) 
                    #plt.imshow(layer)
                    #cv2.imshow("str(i)", layer)
                    st.image(layer)
                    


    elif choice == "Detect Face Components":
        html_temp = """<div style= "background-color: #0a366b;padding:3px 10px; text-align: center; margin-bottom: 15px">
        <h2 style= "color: white; font-weight: 700">Simple Face Components Detection</h2>"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write(" ")

        #upload a new image file for face components
        face_image_file = st.file_uploader("Upload Your Image", type = ['jpg', 'jpeg', 'png'])

        if face_image_file is not None:
            our_face_image = Image.open(face_image_file)
            face_components_type = st.sidebar.radio("Select Face Components", ["Default","Face", "Eye", "Smile"])
            if face_components_type == "Default":
                st.text("Your Uploaded Image")
                st.image(our_face_image)

            elif face_components_type == "Face":
                result_img,result_faces = detect_faces(our_face_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))

            elif face_components_type == "Eye":
                eye_result_img = detect_eyes(our_face_image)
                st.image(eye_result_img)

            elif face_components_type == "Smile":
                smile_result_file = detect_smiles(our_face_image)
                st.image(smile_result_file)

    elif choice == "Color Quantization":
        html_temp = """<div style= "background-color: #0a366b;padding:3px 10px; text-align: center; margin-bottom: 15px">
        <h2 style= "color: white; font-weight: 700">Color Quantization</h2>"""
        st.markdown(html_temp, unsafe_allow_html=True)
        st.write(" ")
        quantization_image_file = st.file_uploader("Upload Image", type = ['jpg', 'jpeg', 'png'])

        if quantization_image_file is not None:
            our_quantization_image = Image.open(quantization_image_file)
            quantization_type = st.sidebar.radio("Select Options", ["Original Image File","Quantized Image"])

            if quantization_type == "Original Image File":
                st.text("Your Uploaded Image")
                st.image(our_quantization_image)
            if quantization_type == "Quantized Image":
                quantization_result_img = color_quantization(our_quantization_image)
                st.image(quantization_result_img)




    elif choice == "About Me":
        about_markdown = read_markdown_file("markdowns/newabout.md")
        st.markdown(about_markdown, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()