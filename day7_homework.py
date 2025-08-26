#每一个batch内找最长padding
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
text_data = load_dataset("allenai/common_gen", split="train")
model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def add_eos_to_examples(example):
    string = ",".join(example['concepts'])  # "ski,mountain,skier"
    example['input_text'] = '%s .' % string
    example['target_text'] = '%s ' % example['target']
    return example
text_data = text_data.map(add_eos_to_examples, batched=False, remove_columns=text_data.column_names)
def custom_collate_fn(example_batch):
    input_texts = [example["input_text"] for example in example_batch]
    target_texts = [example["target_text"] for example in example_batch]
    input_encodings = tokenizer(input_texts, padding=True, truncation=True,return_tensors="pt")#提示词转化为token,保留句子，掩码和token_id#这个时候对input_texts和target_texts按batch最长编码
    target_encodings = tokenizer(target_texts, padding=True,truncation=True,return_tensors="pt").input_ids#句子转化为token，只保留句子，不保留掩码和token_id#maxlength默认是模型的最大长度，True是batch的最大长度
    labels = target_encodings.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": input_encodings['input_ids'],"attention_mask":input_encodings['attention_mask'],"labels":labels}
text_loader = torch.utils.data.DataLoader(text_data,batch_size=2,shuffle=False,num_workers=0,collate_fn=custom_collate_fn)
try:
    for batch in text_loader:
        print(batch)
        print(tokenizer.decode(batch['input_ids'][0]))
        print(tokenizer.decode(batch["labels"][0]))
        break
except Exception as e:
    print("error:", e)
# {'input_ids': tensor([[49406,  3428,   267,  3965,   267, 42585,   269, 49407],[49406,  3428,   267,  3965,   267, 42585,   269, 49407]]),
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]]),
# 'labels': tensor([[49406, 42585, 32843,  1136,   518,  3965,  -100,  -100,  -100,  -100],[49406,   320, 42585,   533, 14400,  1136,   320,  3965,   269,  -100]])}
# <|startoftext|>ski , mountain , skier . <|endoftext|>