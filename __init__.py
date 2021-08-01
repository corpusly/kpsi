# cp from pigai, 21-7-15
import spacy
import pandas as pd

nlp			= spacy.load('en_core_web_sm')
merge_nps	= nlp.create_pipe("merge_noun_chunks")
postag		= lambda snt: pd.DataFrame([ (t.text, t.tag_) for t in nlp(snt)], columns=['word','pos'])
tokenize	= lambda snt: " ".join([t.text for t in nlp(snt) if len(t.text.strip())]).strip()

def parse(snt, merge_np= False):
	doc = nlp(snt)
	if merge_np : merge_nps(doc)
	return pd.DataFrame({'word': [t.text for t in doc], 'tag': [t.tag_ for t in doc],'pos': [t.pos_ for t in doc],'head': [t.head.orth_ for t in doc],'dep': [t.dep_ for t in doc], 'lemma': [t.text.lower() if t.lemma_ == '-PRON-' else t.lemma_ for t in doc],
	'lefts': [ list(t.lefts) for t in doc], 'n_lefts': [ t.n_lefts for t in doc], 'left_edge': [ t.left_edge for t in doc], 'rights': [ list(t.rights) for t in doc], 'n_rights': [ t.n_rights for t in doc], 'right_edge': [ t.right_edge for t in doc],
	'subtree': [ list(t.subtree) for t in doc],'children': [ list(t.children) for t in doc],})

def highlight(snt, merge_np= False,  colors={'ROOT':'red', 'VERB':'orange','ADJ':'green'}, font_size=0):
	doc = nlp(snt)
	if merge_np : merge_nps(doc)
	arr = [ f"<span pos='{t.tag_}'>{t.text.replace(' ','_')}</span>" for t in doc]
	for i, t in enumerate(doc): 
		if t.dep_ == 'ROOT': arr[i] = f"<b><font color='red'>{arr[i]}</font></b>"
		if t.pos_ in colors: arr[i] = f"<font color='{colors[t.pos_]}'>{arr[i]}</font>"
	html =  " ".join(arr) 
	return html if font_size <=0 else f"<div style='font-size:{font_size}px'>{html}</div>"

if __name__ == '__main__': 
	pass