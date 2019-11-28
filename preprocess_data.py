import os
import re
import random
from vars import *
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
    story = re.sub(r"[,;:@#?!&$\-\'\"`]+", " ", story)
    story = " ".join(story.split())
    story = story.lower()
    return story


def process_story(doc: str):
    index = doc.find("@highlight")
    story = doc[:index]
    highlights = doc[index:].split("@highlight")
    highlights = [h.strip() for h in highlights if len(h) > 0]
    highlights = " . ".join(highlights)
    return story, highlights
    # return story, " ".join(["{} {} {}".format(SENTENCE_START, sentence, SENTENCE_END) for sentence in highlights])


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
    with open(filename, 'wb+') as f:
        dump(data, f)


def split_sets(train=.92, filename=filename_data, batch_hack=None) -> None:
    '''split data into train, validation and test sets
    Params:
        - batch_hack: None or int specifying that the length of the dataset should be
          a multiple of the batch_size
    '''
    data_file = os.path.join(data_dir, filename)
    train_file, train_file_hacked, test_file, test_file_hacked = 'train.pkl', 'train_hacked.pkl', 'test.pkl', 'test_hacked.pkl'

    with open(data_file, 'rb') as f:
        data = load(f)
        random.shuffle(data)

        # computing sets indexes
        indexes_train = int(len(data) * train)

        # saving the different files
        if batch_hack:
            train_indexes = (indexes_train // batch_hack) * batch_hack
            test_indexes = ((len(data) - train_indexes) // batch_hack) * batch_hack

            save_pickle(data[:train_indexes], filename=train_file_hacked)
            save_pickle(data[train_indexes: train_indexes + test_indexes], filename=test_file_hacked)
        else:
            save_pickle(data[:indexes_train], filename=train_file)
            save_pickle(data[indexes_train:], filename=test_file)


if __name__ == "__main__":
    dirs = ["data/cnn_stories_tokenized", "data/dm_stories_tokenized"]
    for dir in dirs:
        data = load_stories(dir)
        save_pickle(data, dir.split("/")[1] + "_parsed")

    # split data into train, validation and test sets
    split_sets(batch_hack=32)
