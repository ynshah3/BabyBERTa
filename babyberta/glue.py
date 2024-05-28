import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# Load BabyBERTa tokenizer and model
model_name = "neulab/babyberta"
tokenizer = AutoTokenizer.from_pretrained('./0/run_0/')
model = AutoModelForSequenceClassification.from_pretrained('./0/run_0/', num_labels=2)  # Adjust num_labels for specific tasks

glue_tasks = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
datasets = {task: load_dataset('glue', task) for task in glue_tasks}

results = {}
for task in glue_tasks:
    glue_dataset = datasets[task]
    print(task, glue_dataset)

    def preprocess_function(examples):
        if task in ['cola', 'sst2']:
            return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)
        elif task in ['mrpc', 'stsb', 'rte', 'wnli', ]:
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)
        elif task in ['mnli', 'ax']:
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=128)
        elif task in ['qqp']:
            return tokenizer(examples['question1'], examples['question2'], truncation=True, padding='max_length', max_length=128)
        else:
            return tokenizer(examples['question'], examples['sentence'], truncation=True, padding='max_length', max_length=128)

    encoded_dataset = glue_dataset.map(preprocess_function, batched=True)
    print(encoded_dataset)

    # training_args = TrainingArguments(
    #     output_dir=f'./results/{task}',
    #     logging_dir="./logs",
    #     evaluation_strategy='epoch',
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    # )

    # metric = load_metric("glue", task)

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=encoded_dataset['train'],
    #     eval_dataset=encoded_dataset['validation'],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )

    # trainer.train()
    # result = trainer.evaluate()
    # results[task] = result

print(results)

# # Load GLUE dataset
# task = "mrpc"  # You can change this to other GLUE tasks
# dataset = load_dataset("glue", task)

# # Preprocess the data
# def preprocess_function(examples):
#     return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=128)

# encoded_dataset = dataset.map(preprocess_function, batched=True)

# # Define training arguments
# training_args = TrainingArguments(
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     output_dir="./results",
#     logging_dir="./logs",
# )

# # Load the metric
# metric = load_metric("glue", task)

# # Define the compute_metrics function
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return metric.compute(predictions=predictions, references=labels)

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["validation"],
#     compute_metrics=compute_metrics,
# )

# # Fine-tune the model
# trainer.train()

# # Evaluate the model
# results = trainer.evaluate()

# print(results)
