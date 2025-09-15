import json
import os
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from PIL import Image
import json
import argparse

def read_json(data_file):
    with open(data_file, 'r') as file:
        return json.load(file)
    
def save_json(data, data_file):
    json.dump(data, open(data_file, "w"), indent=2)


MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}


def generate_data(data_path: str, image_head: str):
    json_file = read_json(data_path)
    for data in json_file[:1000]:
        images = [Image.open(os.path.join(image_head,p)) for p in data['images']]
        n_img = len(images)
        yield {
                "images": images,
                "problem": "<image>" *n_img + data["question"],
                "answer": data['answer'], # data["choices"][MAPPING[data["answer"]]] => choice 자체로 변환해서 쓰나봄
        }

def main(image_head, train_path, test_path):
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": train_path, "image_head": image_head})
    # valset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "val")})
    testset = Dataset.from_generator(generate_data, gen_kwargs={"data_path":test_path, "image_head" : image_head})
    dataset = DatasetDict({"train": trainset, "test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub("joyhee/mindcube_rl_1000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mind_cube")
    parser.add_argument("--local_dir", default="/workspace/dataset/MindCube/MindCube/rl_data")
    parser.add_argument("--single_img", action="store_true")
    parser.add_argument("--multi_img", action="store_true")

    args = parser.parse_args()

    if args.dataset == 'mind_cube':
        image_head = '/workspace/dataset/MindCube/MindCube/data'
        if args.single_img:
            question_dir_train = '/workspace/dataset/MindCube/MindCube/data/prompts/mind_cube@train@rawqa_single_image.json'
            question_dir_test = '/workspace/dataset/MindCube/MindCube/data/prompts/mind_cube@rawqa_single_image.json'
        elif args.multi_img:
            question_dir_train = '/workspace/dataset/MindCube/MindCube/data/prompts/mind_cube@train@rawqa_image_list.json'
            question_dir_test = '/workspace/dataset/MindCube/MindCube/data/prompts/mind_cube@test@rawqa_image_list.json'

    main(image_head, question_dir_train, question_dir_test)
