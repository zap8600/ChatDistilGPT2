import math
from datasets import load_dataset
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, pipeline

huggingface_hub.login("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")

huggingface_hub.notebook_login()

tokenizer = AutoTokenizer.from_pretrained("./models")
model = AutoModelForCausalLM.from_pretrained("./models")

eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="my_awesome_eli5_clm-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()

prompt = "Somatic hypermutation allows the immune system to"

generator = pipeline("text-generation", model="my_awesome_eli5_clm-model")
generator(prompt)

tokenizer = AutoTokenizer.from_pretrained("my_awesome_eli5_clm-model")
inputs = tokenizer(prompt, return_tensors="pt").input_ids

model = AutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

tokenizer.batch_decode(outputs, skip_special_tokens=True)
