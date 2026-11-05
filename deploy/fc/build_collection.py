"""Build reusable local Postman requests. Secrets are environment variables only."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
items=[]
def add(name,method,path,status=200,body=None,content_type='application/json',auth=None,extra=()):
    req={'method':method,'url':'{{base_url}}'+path,'header':[]}
    if auth:req['auth']=auth
    if body is not None:
        req['header']=[{'key':'Content-Type','value':content_type}]
        req['body']={'mode':'raw','raw':json.dumps(body,ensure_ascii=False) if not isinstance(body,str) else body}
    lines=[f"pm.test('HTTP {status}', () => pm.response.to.have.status({status}));"]+list(extra)
    items.append({'name':f'{len(items)+1:02d} {name}','request':req,'event':[{'listen':'test','script':{'type':'text/javascript','exec':lines}}]})
add('Service version','GET','/',extra=["pm.test('Router identity', () => pm.expect(pm.response.json().service).eq('rethos-router-fc'));"])
add('Live','GET','/health/live',extra=["pm.test('Alive', () => pm.expect(pm.response.json().status).eq('alive'));"])
add('Ready','GET','/health/ready',extra=["pm.test('Ready', () => pm.expect(pm.response.json().status).eq('ready'));"])
add('OpenAPI','GET','/openapi.json',extra=["pm.test('Business operation', () => pm.expect(pm.response.json().paths['/route']).to.have.property('post'));"])
checks=["pm.test('Exactly three fields', () => pm.expect(Object.keys(pm.response.json()).sort()).deep.eq(['harness','model','skills']));", "pm.test('Valid route', () => {const b=pm.response.json();pm.expect(['codex','claude','dsh']).include(b.harness);pm.expect(['deepseek','glm']).include(b.model);pm.expect(b.skills).to.be.an('array');pm.expect(b.skills.length).at.most(1);});"]
for name,text,model in [
 ('Corpus retrieval','Search only the uploaded fixed paper corpus and provide cited evidence on CRISPR off-target measurement.',None),
 ('Deep research','Research current evidence on sodium-ion battery degradation and synthesize sources.',None),
 ('Research report','Produce a research report covering every rubric item, with methods, evidence, limitations and conclusions.',None),
 ('Scientific execution','Analyze the uploaded CSV in Python, fit a regression and plot confidence intervals.',None),
 ('GLM constraint','Use GLM only. Analyze experimental data and generate statistical plots.','glm'),
 ('DeepSeek speed constraint','Use DeepSeek only. Prioritize low latency. Investigate evidence on battery recycling.','deepseek')]:
    add(name,'POST','/route',body={'request':text},extra=checks+([f"pm.test('Provider preserved', () => pm.expect(pm.response.json().model).eq('{model}'));"] if model else []))
add('Reject blank','POST','/route',400,{'request':' '})
add('Reject extra field','POST','/route',400,{'request':'hello','extra':True})
add('Reject duplicate JSON key','POST','/route',400,'{"request":"a","request":"b"}')
add('Reject wrong content type','POST','/route',415,'hello','text/plain')
add('Reject missing token','GET','/health/live',403,auth={'type':'noauth'})
collection={'info':{'name':'Router Alibaba FC - routing acceptance','schema':'https://schema.getpostman.com/json/collection/v2.1.0/collection.json','description':'One business API, GPU retrieval and one review call. Error cases intentionally expect 4xx. Use independent Router token.'},
 'auth':{'type':'bearer','bearer':[{'key':'token','value':'{{fc_token}}','type':'string'}]},
 'variable':[{'key':'base_url','value':'https://rethos-router-rpboglliga.cn-hangzhou.fcapp.run'},{'key':'fc_token','value':''}],'item':items}
(ROOT/'deploy/fc/router-fc.postman_collection.json').write_text(json.dumps(collection,ensure_ascii=False,indent=2),encoding='utf-8')
print('Collection requests',len(items))
