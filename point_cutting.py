import gradio as gr
from gradio_image_prompter import ImagePrompter
import cv2
from PIL import Image
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = None
scale_factor = 0.4
LOAD_ROOT = os.path.join(os.getcwd(),"<PATH...>")
SAVE_ROOT = os.path.join(os.getcwd(),"<PATH...>")

# 이미지 자르기 및 저장 함수
def start(prompt:dict, root:str, name:str, line_path:str, color_path:str, base_path:str):
    gr.Warning(f"{name} 시작", duration=1.5)

    # 이미지 읽기
    image = Image.open(os.path.join(line_path, name))
    color = Image.open(os.path.join(color_path, name))
    base = Image.open(os.path.join(base_path, name))

    # 점 또는 박스 구분
    # 점은 prompt['points'][3] == 1.0
    # 박스는 prompt['points'][3] == 2.0
    # 자를 포인트 계산
    boxes = []
    tempbox = []
    for coord in prompt['points']:
        if coord[2] == 2.0:
            boxes.append(
                tuple([
                    float(coord[0] / scale_factor),
                    float(coord[1] / scale_factor),
                    float(coord[3] / scale_factor),
                    float(coord[4] / scale_factor)
                ])
            )
        elif coord[2] == 1.0:
            if tempbox != []:
                tempbox.append(float(image.width))
                tempbox.append(float(coord[1] / scale_factor))
                boxes.append(tuple(tempbox))
                tempbox = []
            else:
                tempbox.append(float(0))
                tempbox.append(float(coord[1] / scale_factor))
    

    # 디렉토리 생성
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "밑색"), exist_ok=True)
    os.makedirs(os.path.join(root, "선화"), exist_ok=True)
    os.makedirs(os.path.join(root, "채색"), exist_ok=True)

    for idx,box in enumerate(boxes):
        cut_img = image.crop(box=box)
        cut_img.save(f"{root}선화/{name}-{idx}.jpg")
        cut_color = color.crop(box=box)
        cut_color.save(f"{root}채색/{name}-{idx}.jpg")
        cut_base = base.crop(box=box)
        cut_base.save(f"{root}밑색/{name}-{idx}.jpg")

    gr.Warning(f"{name} 완료", duration=1.5)

# 이미지 리사이즈 함수
def resize_image(image):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width,new_height),resample=Image.Resampling.LANCZOS)
    return {"image": resized_image, "points": []}

# 파일 리스트 생성 함수
def get_file_list(directory:str):
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f!=".DS_Store"])
    return files

def up(name:str):
    path = os.path.join(LOAD_ROOT,"<선화>",name)
    return resize_image(Image.open(path))

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            line_path = gr.Textbox(label="선화 경로", value=os.path.join(LOAD_ROOT,"<선화>"))
            base_path = gr.Textbox(label="밑색 경로", value=os.path.join(LOAD_ROOT,"<밑색>"))
            color_path = gr.Textbox(label="채색 경로", value=os.path.join(LOAD_ROOT,"<채색>"))
            root_path = gr.Textbox(label="저장 위치", value=SAVE_ROOT)
            
            # 파일 선택 드롭다운 추가
            file_list = get_file_list("./train_data/스피노프 백작가/5~6화/선/")  # 실제 선화 경로에서 파일 목록 가져오기
            name = gr.Dropdown(choices=file_list, label="파일 선택", value=None)

        with gr.Column():
            img_prompter = ImagePrompter(label="이미지 프롬프트", type="pil", image_mode="RGB", width=500)
            name.change(fn=up,inputs=name,outputs=img_prompter)

            btn_cutting = gr.Button("✂️ 자르기")
            btn_cutting.click(
                fn=start,
                inputs=[img_prompter, root_path, name,line_path, color_path, base_path],
                outputs=[]
            )

demo.launch()
