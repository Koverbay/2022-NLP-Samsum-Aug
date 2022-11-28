# Generates a json file with lists of entities from the dataset for the 12 entity types used by spacy.
import spacy
import argparse
import json
from datasets import load_dataset


def main(args):
    # Load the dataset and spacy model.
    samsum = load_dataset('samsum', split=args.split)
    nlp = spacy.load("en_core_web_lg")

    # Create the dictionary to store the entities.
    entity_labels = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
    entities = {}
    for entity_label in entity_labels:
        entities[entity_label] = []

    # For each item in dataset, find and save entities using spacy.
    for item in samsum:
        text = item['summary']
        doc = nlp(text)

        for ent in doc.ents:
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)

    # Save the final entity dictionary to a json file.
    with open(args.filename, 'w') as f:
        json.dump(entities, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default="../data/entities.json")
    parser.add_argument('-sp', '--split', choices=["test", "train", "val"], default="train")
    args = parser.parse_args()
    main(args)