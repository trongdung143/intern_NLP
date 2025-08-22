import spacy
import random
from processing_data import prcesssing_data
from spacy.util import minibatch, compounding
from spacy.training import Example

TRAIN_DATA = prcesssing_data()
nlp = spacy.load("en_core_web_lg")


if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")


for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    optimizer.L2 = 1e-6
    optimizer.L2_is_weight_decay = True

    best_loss = float("inf")
    best_model = None
    patience = 5
    wait = 0

    epochs = 50
    for epoch in range(epochs):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 16.0, 1.07))
        drop = 0.2 + (epoch / epochs) * 0.1
        lr = 0.001 - (epoch / epochs) * (0.001 - 0.0003)
        optimizer.learn_rate = lr

        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)

        current_loss = losses.get("ner", 0.0)
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")

        if current_loss < best_loss:
            best_loss = current_loss
            nlp.to_disk("./model")
            wait = 0
            print("Saved new best model.")
        else:
            wait += 1
            print(f"No improvement. Patience: {wait}/{patience}")

        if wait >= patience:
            print("Early stopping triggered.")
            break
