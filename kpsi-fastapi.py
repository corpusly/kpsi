#  uvicorn kpsi-fastapi:app --host 0.0.0.0 --port 7080 --reload 
import uvicorn , time, json, pymysql 
from collections import defaultdict,Counter
from fastapi import FastAPI
from pydantic import BaseModel
import time,pymysql, json,math
import numpy as np
import pandas as pd

my_conn = pymysql.connect(host='192.168.1.24',port=3306,user='root',password='cikuutest!',db='kpsi')
cursor = my_conn.cursor()  #∑
rows = lambda sql: list((cursor.execute(sql), cursor.fetchall())[1])
getsi = lambda s , corpus='dic', defa=0.1:  ( res:=rows(f"select i from {corpus} where s = '{s}'"),  res[0][0] if len(res) > 0 else defa  )[1]
keyness = lambda a,b,c,d : round(a * math.log(a*(c+d)/(c*(a+b))) +    b * math.log(b*(c+d)/(d*(a+b)))  if a/c > b/d else  -  a * math.log(a*(c+d)/(c*(a+b)))  -  b * math.log(b*(c+d)/(d*(a+b))), 1)
pos_attr = lambda pos='VERB': [ row[0] for row in  rows(f"select attr from pos_attr where pos  = '{pos}'")]
wordlist = lambda pos='VERB': [ row[0] for row in  rows(f"select substring_index(k,':',-1) from wordlist where k  like '{pos}:%'")]
kn = lambda s1, s2, cp1, cp2:  keyness( getsi(s1, cp1), getsi(s1, cp2), getsi(s2,cp1), getsi(s2, cp2) )

getlist = lambda pat="~dobj_VERB_NOUN:knowledge:%" , cp='dic':  [ row[0] for row in  rows(f"select substring_index(s,':',-1) from {cp} where s like '{pat}'")]
getunion = lambda pat="~dobj_VERB_NOUN:knowledge:%" , cp1='dic', cp2='clec':  [ row[0] for row in  rows(f"select substring_index(s,':',-1) from {cp1} where s like '{pat}' union select substring_index(s,':',-1) from {cp2} where s like '{pat}'")]
getmap = lambda pat="~dobj_VERB_NOUN:knowledge:%" , cp='dic':  dict(rows(f"select substring_index(s,':',-1), i from {cp} where s like '{pat}'"))

app = FastAPI()

@app.get('/kpsi/sisql')
def sisql(sql:str="select * from bnc where s like 'work:VERB:VB%'  or s in ('VERB:work')", divby:str=None): 
	''' rows("select * from bnc where s like 'work:VERB:VB%'  or s in ('VERB:work')") '''
	res  = rows(sql)
	if not divby : return res    
	res = dict(res)    
	divsum  = res.get(divby, sum( [v for k,v in res.items()] ) )   
	return [(k,v, v/divsum)  for k,v in res.items()]

@app.get('/kpsi/attr')
def attr_keyness(w:str="consider", pos:str="VERB", cp1:str="sino", cp2:str="bnc", ascending:bool=True): 
	''' consider:VERB:*  attr list  -> keyness '''
	df = pd.DataFrame([ (attr, kn(f'{w}:{pos}:{attr}', '{pos}:{w}', cp1,cp2)) for attr in pos_attr(pos) ], columns=['attr','kn'])
	df = df.sort_values(by="kn" , ascending=ascending)
	return df.to_numpy().tolist()

from cikuu.dic.lemma_scale import lemma_scale
@app.get('/kpsi/scale_keyness')
def kpsi_scale_keyness(scale_beg:float=5.0, scale_end:float=6.0, cp1:str="sino", cp2:str="bnc", ascending:bool=True): 
	''' lemma list  -> keyness '''
	df = pd.DataFrame([ (w, kn(f'LEM:{w}', 'sum:LEM', cp1,cp2)) for w, s in lemma_scale.items() if s >= scale_beg and s <= scale_end ], columns=['attr','kn'])
	df = df.sort_values(by="kn" , ascending=ascending)
	return df.to_numpy().tolist()

@app.get('/kpsi/attr_keyness_of_one_corpus')
def attr_keyness_of_one_corpus(w:str="work", attr:str="VBG", pos:str="VERB", cp:str="bnc"): 
	''' keyness( "consider:VERB:VBG", "sum:VBG",  "VERB:consider", "sum:VERB" ) '''
	return keyness( getsi(f'{w}:{pos}:{attr}', cp), getsi(f"sum:{attr}", cp), getsi(f"{pos}:{w}",cp), getsi(f"sum:{pos}", cp) )

@app.get('/kpsi/pos_attr_keyness_of_one_corpus')
def pos_attr_keyness_of_one_corpus(attr:str="VBG", pos:str="VERB", cp:str="bnc", ascending:bool=True): 
	b = getsi(f"sum:{attr}", cp)
	d = getsi(f"sum:{pos}", cp)    
	df = pd.DataFrame([ (w,  keyness( getsi(f'{w}:{pos}:{attr}', cp), b, getsi(f"{pos}:{w}",cp), d ) )  for w in wordlist(pos) if w.isalpha() ], columns=['word', 'kn'])
	df = df.sort_values(by="kn" , ascending=ascending)    
	return df.to_numpy().tolist()

@app.get('/kpsi/triple_keyness_of_two_corpus')
def triple_keyness_of_two_corpus(pat:str="~dobj_VERB_NOUN:knowledge", cp1:str="clec", cp2:str='dic', ascending:bool=True): 
	words = getunion(f"{pat}:%",cp1, cp2)
	clec = getmap(f"{pat}:%",cp1)
	dic = getmap(f"{pat}:%", cp2)
	c = getsi(f"#{pat}", cp1)
	d = getsi(f"#{pat}", cp2)    
	df = pd.DataFrame({'words': words, cp1: [ clec.get(w, 0) for w in words], cp2: [ dic.get(w, 0) for w in words], 'keyness':[ keyness(clec.get(w,0.1), dic.get(w,0.1), c, d)  for w in words]})
	df = df.sort_values(by="keyness" , ascending=ascending)
	return df.to_numpy().tolist()

@app.get('/kpsi/chunk_keyness_of_two_corpus')
def chunk_keyness_of_two_corpus(w:str="book", pos:str='NOUN', seg:str='np', cp1:str="clec", cp2:str='dic', ascending:bool=True): 
	words = getunion(f"{w}:{seg}:%",cp1,cp2)
	clec = getmap(f"{w}:{seg}:%",cp1)
	dic = getmap(f"{w}:{seg}:%",cp2)
	c = getsi(f"{w}:{pos}:{seg}", cp1)
	d = getsi(f"{w}:{pos}:{seg}", cp2)
	df = pd.DataFrame({'word': words, cp1: [ clec.get(w, 0) for w in words], cp2: [ dic.get(w, 0) for w in words], 'keyness':[ keyness(clec.get(w,0.1), dic.get(w,0.1), c, d)  for w in words]})
	df = df.sort_values(by="keyness" , ascending=True)
	return df.to_numpy().tolist()

@app.get('/kpsi/attr_keyness_of_two_corpus')
def attr_keyness_of_two_corpus(w:str="work", attr:str="VBG", pos:str="VERB", cp1:str="sino", cp2:str="bnc"): 
	return kn(f'{w}:{pos}:{attr}', f'{pos}:{w}', cp1, cp2)

from fastapi.responses import HTMLResponse
@app.get('/')
def home(): return HTMLResponse(content=f"<h2> mariadb-based kpsi http api </h2> <a href='/docs'> docs </a> | <a href='/redoc'> redoc </a> <hr> <a href='/kpsi/sisql?sql=select%20%2A%20from%20bnc%20where%20s%20like%20%27sum%3AVB%25%27%20%20or%20s%20in%20%28%27sum%3AVERB%27%29&divby=sum%3AVERB'>sisql sample</a> <br>2021-7-14")

if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port=7080)