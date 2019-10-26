import os
import re
from progress.bar import Bar
from typing import List, Dict, Union
from pickle import dump

strip_regex = "|".join(['LRB', 'RRB', 'LSB', 'RSB'])


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


def save_pickle(filename: str, data: List[Dict[str, Union[str, str]]]) -> None:
    with open(filename, 'wb') as f:
        dump(data, f)
    print("{} created!\n".format(filename))


if __name__ == "__main__":
    dirs = ["data/cnn_stories_tokenized", "data/dm_stories_tokenized"]
    for dir in dirs:
        data = load_stories(dir)
        save_pickle(dir + ".pkl", data)
