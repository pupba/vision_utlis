from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm


ROOT =""
TOON = [""]


SAVE_ROOT = "cn_data/"

def resizing(image:Image, target_size=(1024,1024)):
    w,h = image.size
    temp_size = (w,w) if w > h else (h,h)

    pad = Image.new("RGB",temp_size,color=(255,255,255))

    if w > h:
        # 가로가 더 긴 경우
        pad.paste(image, (0, (temp_size[1] - h) // 2))
    else:
        # 세로가 더 긴 경우
        pad.paste(image, ((temp_size[0] - w) // 2, 0))
    
    # 최종 이미지를 타겟 사이즈로 리사이즈
    resized_image = pad.resize(target_size, Image.Resampling.LANCZOS)
    
    return resized_image

def toCanny(image:Image):
    img = np.array(image.convert("L"))

    canny = cv2.Canny(img,100,200)

    return canny

def save_prompt(fname:str):
    prompt = "a photo of webtoon cut image,character,no line art image,webtoon frames,white background"
    with open(fname,"w+") as f:
        f.write(prompt)

def makeDataset(folders:list,fname:int)->int:
    for fd in tqdm(folders,desc="process..."):
        path = ROOT+TOON[0]+fd+"/"
        imgs = [(Image.open(path+f"채색/{i}").convert("RGB"),Image.open(path+f"밑색/{i}").convert("RGB")) for i in os.listdir(path+"채색/") if i.endswith(".jpg")]

        for c,u in imgs:
            # 1. 이미지 1024,1024 리사이즈
            rc = resizing(c)
            # 2. Canny (Input 1)
            crc = toCanny(rc)
            # 3. prompt (Input 2)
            save_prompt(SAVE_ROOT+f"prompt/{str(fname).zfill(3)}.txt")
            # 4. Undertone (Output)
            ru = resizing(u)
            # 5. save imgs
            cv2.imwrite(SAVE_ROOT+f"conditioning_image/{str(fname).zfill(3)}.jpg",crc) # Canny
            ru.save(SAVE_ROOT+f"color/{str(fname).zfill(3)}.jpg")# ut
            fname +=1

    return fname

if __name__ == "__main__":

    f1 = [i for i in os.listdir(ROOT+TOON[0]) if i!=".DS_Store"]
    f2 = [i for i in os.listdir(ROOT+TOON[1]) if i!=".DS_Store"]

    fname = 0
    fname = makeDataset(f1,fname)
    fname = makeDataset(f2,fname)
