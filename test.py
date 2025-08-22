import spacy

trained_nlp = spacy.load("./model")

test = "TRẦN QUỐC NGUYỆN F21,C3,Tan Thoi Nhat, District 12, HCM city E-mail : nguyenpc2010@gmail.com "

doc = trained_nlp(test)
for ent in doc.ents:
    print(f"{ent.label_}: {ent.text}")
