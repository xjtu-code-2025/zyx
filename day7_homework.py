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
def convert_to_features(example_batch):
    input_encodings = tokenizer(example_batch['input_text'], padding="max_length", max_length=16, truncation=True,
                                return_tensors="pt")#提示词转化为token,保留句子，掩码和token_id
    target_encodings = tokenizer(example_batch['target_text'], padding="max_length", max_length=16, truncation=True,
                                 return_tensors="pt").input_ids#句子转化为token，只保留句子，不保留掩码和token_id
    labels_with_ignore_index = []
    for labels_example in target_encodings:#一个batch的句子的token,labels_example表示一个句子的token
        labels_example = [label if label != 0 else -100 for label in labels_example]#token=0就是-100,不然不变
        labels_with_ignore_index.append(labels_example)#二维列表，每一个元素是一个token的列表
    encodings = {
        'input_ids': input_encodings['input_ids'],#提示词的token
        'attention_mask': input_encodings['attention_mask'],#提示词的掩码
        'labels': labels_with_ignore_index#句子的token
    }
    return encodings
text_data = text_data.map(add_eos_to_examples, batched=False, remove_columns=text_data.column_names)
print(text_data[0])
text_data = text_data.map(convert_to_features, batched=True, remove_columns=text_data.column_names)
print(text_data[0])
def custom_collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        input_ids += [item["input_ids"]]
        attention_mask += [item["attention_mask"]]
        labels += [item["labels"]]
    return {"input_ids":input_ids,"attention_mask":attention_mask,"labels":labels}
text_loader = torch.utils.data.DataLoader(text_data,batch_size=10,shuffle=True,num_workers=0,collate_fn=custom_collate_fn)
try:
    for batch in text_loader:
        print(batch)
        break
except Exception as e:
    print("error:", e)
