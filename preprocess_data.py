import os
import re
import random
from itertools import repeat
from progress.bar import Bar
from typing import List, Dict, Union
from pickle import dump, load

strip_regex = "|".join(['LRB', 'RRB', 'LSB', 'RSB'])
filename_data = "data.pkl"
data_dir = "data/"


def load_file(filename: str) -> str:
    with open(filename, 'r', encoding='utf8') as f:
        text = f.read()
        return text


def clean_story(story: str) -> str:
    story = re.sub(strip_regex, " ", story)
    story = re.sub(r"[,;@#?!&$\-\'\"\`]+\ *", " ", story)
    story = " ".join(story.split())
    story = story.lower()
    return story


def process_story(doc: str):
    index = doc.find("@highlight")
    story = doc[:index]
    highlights = doc[index:].split("@highlight")
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, " . ".join(highlights)


def load_stories(dir_name: str) -> List[Dict[str, Union[str, str]]]:
    data = []
    list_files = os.listdir(dir_name)
    bar = Bar("Processing data", max=len(list_files))
    for file in list_files:
        filename = os.path.join(dir_name, file)
        doc = load_file(filename)
        story, highlights = process_story(doc)
        story = clean_story(story)
        data.append({'story': story,
                     "highlights": highlights})
        bar.next()
    bar.finish()
    return data


def save_pickle(data: List[Dict[str, Union[str, str]]], filename=filename_data) -> None:
    filename = os.path.join(data_dir, filename)
    with open(filename, 'ab') as f:
        dump(data, f)


def split_sets(train=.92, validation=.05, filename=filename_data) -> None:
    '''split data into train, validation and test sets'''
    data_file = os.path.join(data_dir, filename)
    train_file, test_file, valid_file = ['train.pkl', 'test.pkl', 'validation.pkl']

    with open(data_file, 'rb') as f:
        data = load(f)
        random.shuffle(data)

        # computing sets indexes
        indexes_train = int(len(data) * train)
        indexes_validation = indexes_train + int(len(data) * validation)

        # saving the different files
        save_pickle(data[:indexes_train], filename=train_file)  # creating train set
        save_pickle(data[indexes_train:indexes_validation], filename=valid_file)  # creating validation set
        save_pickle(data[indexes_validation:], filename=test_file)  # creating test set


if __name__ == "__main__":
    dirs = ["data/cnn_stories_tokenized", "data/dm_stories_tokenized"]
    for dir in dirs:
        data = load_stories(dir)
        save_pickle(data)

    # split data into train, validation and test sets
    split_sets()
