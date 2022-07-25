import spacy

from spacy import displacy


nlp = spacy.load("zh_core_web_sm")

s = "小米董事长叶凡决定投资华为。在2002年，他还创作了<遮天>。"

doc = nlp(s)

for i in doc.sents:
    print(i)


print([(w.text, w.dep_) for w in doc])

# 可视化依存关系
html_str = displacy.render(doc, style="dep")

with open("test.html", "w", encoding="utf8") as f:
    f.write(html_str)