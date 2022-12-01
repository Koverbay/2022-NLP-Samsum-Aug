# Creates and saves a new dataset object with entities in the summaries swapped with other entities from the dataset.
# This will be used as a 'negative' dataset with the prompt 'Generate an incorrect summary.'
import argparse
import spacy
import random
import json
from datasets import load_dataset
import colorful as cf

entities = {}
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('merge_entities')

def main(args):

    # Load dataset, spacy model, and the entity lists.
    samsum = load_dataset('samsum', split=args.split)
    with open (args.entities, 'r') as f:
        global entities
        entities = json.load(f) 


    # For each item in the dataset, swap the entities with random other entities.
    # Append the prompt 'Generate a false summary.' to the input.
    # Save these new items to the new dataset.
    samsum_swapped = samsum.map(replace_entity)

    # Save the new dataset object.
    samsum_swapped.save_to_disk(args.savepath)

def replace_entity(item):
    text = item['summary']
    doc = nlp(text)

    new_sum = " ".join([t.text if not t.ent_type_ else random_ent(t.ent_type_) for t in doc])
    item['summary'] = new_sum
    item['dialogue'] = "Generate an incorrect summary. " + item['dialogue']
    return item


def random_ent(entity_type):
    l = len(entities[entity_type])
    i = random.randint(0,l-1)
    return entities[entity_type][i]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savepath', default="../data/samsum-entity-swap")
    parser.add_argument('-e', '--entities', default="../data/entities.json")
    parser.add_argument('-sp', '--split', choices=["test", "train", "val"], default="train")
    args = parser.parse_args()
    main(args)