import json
import os
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from dataclasses import dataclass
from PIL import Image
import json
import argparse
import re
import math

from huggingface_hub import login

def read_json(data_file):
    with open(data_file, 'r') as file:
        return json.load(file)
    
def save_json(data, data_file):
    json.dump(data, open(data_file, "w"), indent=2)

@dataclass
class ImageProcessor:
    max_pixels: int
    min_pixels: int

    def __call__(self, image: Image.Image):
        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


sft_question_multi = """
Question: 
Based on the image, where is the <target> located relative to the <anchor> when viewed from the front side of the <image_order> image?
Options:
A. <option_A> 
B. <option_B> 
C. <option_C> 
D. <option_D>
"""

block_question = """
Here are the images of top view: <image> front view: <image> side view: <image> of a three-dimensional structure.
Question: 
Which of the following could be the actual three-dimensional structure? 
Options:
A. <image> 
B. <image> 
C. <image> 
D. <image>
Please select the correct answer from the following options.
"""


block_question_c2 = """
Here are the images of top view: <image> front view: <image> side view: <image> of a three-dimensional structure.
Question: 
Which of the following could be the actual three-dimensional structure? 
Options:
A. <image> 
B. <image> 
Please select the correct answer from the following options.
"""


def generate_data(data_path: str, image_head: str, data_name:
     str):
    json_file = read_json(data_path)
    processor = ImageProcessor(max_pixels=256 * 256, min_pixels=64 * 64)
    processor_small = ImageProcessor(max_pixels= 64 * 64, min_pixels=32 * 32)
    for data in json_file[:1000]:
        if isinstance(image_head, dict) : cur_image_head = image_head[data['question_type']]
        else: cur_image_head = image_head

        if 'question' in data:
            if isinstance(data['question'], list):
                images = [processor(Image.open(os.path.join(cur_image_head,p))) for p in data['question'] if bool(re.search(r'\.(png|jpg)\b', p, re.IGNORECASE))]
                question = ''.join(["<image>" if bool(re.search(r'\.(png|jpg)\b', p, re.IGNORECASE)) else p for p in data['question']])
                yield {
                        "images": images,
                        "problem": question,
                        "answer": data['answer'], # data["choices"][MAPPING[data["answer"]]] => choice 자체로 변환해서 쓰나봄
                }

        elif 'images' in data:
            images = []
            for p in data['images']:
                img = Image.open(os.path.join(cur_image_head,p))
                img.load()
                images.append(processor_small(img))
                img.close()
            #images = [Image.open(os.path.join(cur_image_head,p)) for p in data['images']]
            n_img = len(images)
            if 'question' not in data:
                if 'sft_multi' in data_name:
                    image_order = ["first", "second", "third"][data["viewpoint"]]
                    data['question'] = sft_question_multi.replace('<target>', data['target'])\
                                                        .replace("<anchor>", data['anchor'])\
                                                        .replace("<image_order>", image_order)\
                                                        .replace('<option_A>', data['A'])\
                                                        .replace('<option_B>', data['B'])\
                                                        .replace('<option_C>', data['C'])\
                                                        .replace('<option_D>', data['D'])
                    yield {
                            "images": images,
                            "problem": "<image>" *n_img + data["question"],
                            "answer": data['answer'], # data["choices"][MAPPING[data["answer"]]] => choice 자체로 변환해서 쓰나봄
                    }
                elif 'block' in data_name:
                    for idx in ['A','B','C','D']:
                        if idx in data:
                            img = Image.open(os.path.join(cur_image_head,data[idx]))
                            img.load()
                            images.append(processor(img))
                            img.close()
                    if 'c2' in data_name:
                        data['question'] = block_question_c2
                    else:
                        data['question'] = block_question
                    
                    yield {
                            "images": images,
                            "problem": data["question"],
                            "answer": data['answer'], # data["choices"][MAPPING[data["answer"]]] => choice 자체로 변환해서 쓰나봄
                    }
            else:
                yield {
                            "images": images,
                            "problem": "<image>" *n_img + data["question"],
                            "answer": data['answer'], # data["choices"][MAPPING[data["answer"]]] => choice 자체로 변환해서 쓰나봄
                    }

def main(image_head, train_path, test_path, data_name):
    if train_path:
        trainset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": train_path, "image_head": image_head, "data_name" : data_name})
    # valset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "val")})
    if test_path:
        testset = Dataset.from_generator(generate_data, gen_kwargs={"data_path":test_path, "image_head" : image_head, "data_name" : data_name})
    
    if (train_path != None) & (test_path != None):
        dataset = DatasetDict({"train": trainset, "test": testset}).cast_column("images", Sequence(ImageData()))
    elif test_path:
        dataset = DatasetDict({"test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub(f"joyhee/{data_name}_1000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mind_cube")
    parser.add_argument("--local_dir", default="/workspace/dataset/MindCube/MindCube/rl_data")
    parser.add_argument("--single_img", action="store_true")
    parser.add_argument("--multi_img", action="store_true")

    args = parser.parse_args()

    if args.dataset == 'mind_cube':
        image_head = '/home/server38/sohee_workspace/SpatialReasoning/dataset/MindCube/MindCube/data'
        if args.single_img:
            question_dir_train = '/home/server38/sohee_workspace/SpatialReasoning/dataset/MindCube/MindCube/data/prompts/mind_cube@train@rawqa_single_image.json'
            question_dir_test = '/home/server38/sohee_workspace/SpatialReasoning/dataset/MindCube/MindCube/data/prompts/mind_cube@rawqa_single_image.json'
        elif args.multi_img:
            question_dir_train = '/home/server38/sohee_workspace/SpatialReasoning/dataset/MindCube/MindCube/data/prompts/mind_cube@train@rawqa_image_list.json'
            question_dir_test = '/home/server38/sohee_workspace/SpatialReasoning/dataset/MindCube/MindCube/data/prompts/mind_cube@test@rawqa_image_list.json'

    elif args.dataset == 'sft_multi_v2':
        image_head = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/image'
        question_dir_train = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/assets/{args.dataset}_train.json'
        question_dir_test = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/assets/{args.dataset}_test.json'

    elif 'blocks' in args.dataset:
        image_head = f'/home/server38/soohyun_workspace/spatial_reasoning/data'
        question_dir_train = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/assets/{args.dataset}_train.json'
        question_dir_test = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/assets/{args.dataset}_test.json'

    elif args.dataset == 'rl_SR':
        image_head = {
            "rl_SR1" : '/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/image',
            "rl_SR2" : '/home/server38/soohyun_workspace/spatial_reasoning/data',
            "rl_SR3" : '/home/server38/sohee_workspace/SpatialReasoning/dataset/SFT_data/image',
            "rl_SR4" : '/home/server38/soohyun_workspace/spatial_reasoning/data',
            "rl_SR5" : '/home/server38/sohee_workspace/SpatialReasoning/dataset/building_blocks/image'
        }
        question_dir_train = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/RL_data/assets/assets_{args.dataset}_train.json'
        question_dir_test = f'/home/server38/sohee_workspace/SpatialReasoning/dataset/RL_data/assets/assets_{args.dataset}_test.json'

    main(image_head, question_dir_train, question_dir_test, args.dataset)
