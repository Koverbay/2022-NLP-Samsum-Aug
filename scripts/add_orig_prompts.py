# Creates and saves a new dataset object with the prompt "Generate a correct summary" prepended to the original
# SamSUM dataset input.
import argparse
import json
from datasets import load_dataset
import colorful as cf



def main(args):

    # Load dataset.
    samsum = load_dataset('samsum', split=args.split)

    # For each item in the dataset, add the prompt "Generate a correct summary" to the input dialogue.
    print("Applying transformations to dataset...")
    samsum_prompt = samsum.map(add_prompt)

    # Save the new dataset object.
    print(f"Saving new dataset at {args.savepath}.")
    samsum_prompt.save_to_disk(args.savepath)

def add_prompt(item):
    item['dialogue'] = "Generate a correct summary. " + item['dialogue']
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savepath', default="../data/samsum-orig-prompt")
    parser.add_argument('-sp', '--split', choices=["test", "train", "val"], default="train")
    args = parser.parse_args()
    main(args)