# Data Augmentation for Dialogue Summarization
## SNU 2022 Fall NLP term project

## Relevant files
### SamSUM dataset
- imported using the datasets library provided by [huggingface datasets](https://huggingface.co/datasets/samsum)

### Files for data loading and entity-replacement (data augmentation)
- add_orig_prompts.py 
- generate_entity_lists.py 
- replace_entities.py
- augmented data in `data` folder

### Files to fine-tune the T5-large language model
- pretraine model from [huggingface models](https://huggingface.co/t5-large)
- train_t5.py
