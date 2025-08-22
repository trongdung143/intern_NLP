import spacy


def predict(text):
    model = spacy.load("./model/ner/")
    doc = model(text)
    result = {"name": None, "email": None}
    for ent in doc.ents:
        if ent.label_ == "NAME":
            result["name"] = ent.text
        elif ent.label_ == "EMAIL":
            result["email"] = ent.text
    return result
