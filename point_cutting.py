import gradio as gr
from gradio_image_prompter import ImagePrompter
import cv2
from PIL import Image
import os
import numpy as np
Image.MAX_IMAGE_PIXELS = None

scale_factor = 0.4

def start(prompt,root,name,l,c,ut):
    gr.Warning(f"{name} start")
    image = cv2.imread(l+name)
    c = cv2.imread(c+name)
    ut = cv2.imread(ut+name)

    # x,y
    points = [(coord[0],coord[1]) for coord in prompt['points']]
    for idx in range(len(points)):
        if idx == 0:
            start = 0
            end = points[idx][1]
        else:
            start = points[idx-1][1]
            end = points[idx][1]

        start_y = int(start / scale_factor)
        end_y = int(end / scale_factor)

        os.makedirs(root,exist_ok=True)
        os.makedirs(root+"밑색/",exist_ok=True)
        os.makedirs(root+"선화/",exist_ok=True)
        os.makedirs(root+"채색/",exist_ok=True)

        # cut_img = np.array(prompt['image'])[start_y:end_y,:]
        cut_img = image[start_y:end_y,:]
        cv2.imwrite(f"{root}선화/{name}-{idx}.jpg",cut_img)
        cut_c = c[start_y:end_y,:]
        cv2.imwrite(f"{root}채색/{name}-{idx}.jpg",cut_c)
        cut_ut = ut[start_y:end_y,:]
        cv2.imwrite(f"{root}밑색/{name}-{idx}.jpg",cut_ut)

    gr.Warning(f"{name} End")

def resize_image(image):
    # 이미지를 축소 비율에 맞춰 리사이즈
    new_width = int(image['image'].shape[1] * scale_factor)
    new_height = int(image['image'].shape[0] * scale_factor)
    return {
        "image":cv2.resize(image['image'], (new_width, new_height),interpolation=cv2.INTER_LANCZOS4),
        "points":image['points']
    }

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            l = gr.Textbox(label="선화 경로",value="line/")
            ut = gr.Textbox(label="밑색 경로",value="ut/")
            c = gr.Textbox(label="채색 경로",value="color/")
            root = gr.Textbox(label="저장 위치",value="data/")
            name = gr.Textbox(label="파일 이름")
            
        with gr.Column():
            img = ImagePrompter(label="image",type="numpy",image_mode="RGB",width=500)
            # 이미지를 축소하여 프롬프트 표시
            img.upload(fn=resize_image,inputs=img,outputs=img)
            btn2 = gr.Button("✂️ cutting")
    btn2.click(fn=start,inputs=[img,root,name,l,c,ut])

demo.launch()
