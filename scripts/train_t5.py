# script to fine tune t5-large model on augmented samsum

import torch, os
import datasets
from transformers import T5Tokenizer, T5Model
from transformers import DataCollatorForSeq2Seq, get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm


def data_tokenization(samples):
    model_inputs = tokenizer(samples['dialogue'], max_length=max_doc_len, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(samples['summary'], max_length=max_sum_len, truncation=True)
        
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


split = "train"
data = datasets.load_from_disk("../data/samsum-orig-prompt")
max_doc_len = 512
max_sum_len = 128

tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5Model.from_pretrained("t5-large")

tokenized_dataset = data.map(data_tokenization, batched=True)
tokenized_dataset.set_format('torch')
tokenized_dataset = tokenized_dataset.remove_columns(data_train['id'].column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


train_dataloader = DataLoader(
    tokenized_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
)

eval_dataloader = DataLoader(
    tokenized_dataset['validation'],
    collate_fn=data_collator,
    batch_size=4,
)




tokenizer = AutoTokenizer.from_pretrained("henryu-lin/t5-large-samsum-deepspeed")
model = AutoModelForSeq2SeqLM.from_pretrained("henryu-lin/t5-large-samsum-deepspeed")
optimizer = AdamW(model.parameters(), lr=2e-5)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)


epochs = 3
update_step_per_epoch = len(train_dataloader)
training_steps = epochs * update_step_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_steps,
)


out_dir = './t5-samsum_tuned'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

progress_bar = tqdm(range(training_steps))

for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            
            labels = batch["labels"]
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )
            
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.pip(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)
    
    result = rouge_score.compute()
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    
    print(f'Epoch {epoch}:', result)
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(out_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(out_dir)