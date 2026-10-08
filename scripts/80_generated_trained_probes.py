#!/usr/bin/env python
"""Train on neutral generated chains; select on disjoint validation and test transfer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cueconf.config import DATA_DIR, OUT_DIR, model_entry
from cueconf.followups import value_boundary
from cueconf.generator import Instance, read_jsonl
from cueconf.prompts import assert_before_value, build_prompt, make_demos
from cueconf.round6 import computation_key, generated_cohorts

BASE=OUT_DIR/'round6_generated'


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def prepare():
    source=read_jsonl(DATA_DIR/'L3/probe_train_neutral.jsonl')
    tests=read_jsonl(DATA_DIR/'L3/test_sets.jsonl')
    demonstrations=[x for seed in (7,11,13) for x in make_demos(3,'word',seed=seed)]
    tr,va=generated_cohorts(source,tests,demonstrations)
    data={'train':[asdict(x) for x in tr],'validation':[asdict(x) for x in va],
          'n_train_keys':len({computation_key(x) for x in tr}),
          'n_validation_keys':len({computation_key(x) for x in va}),
          'train_validation_overlap':0,'test_overlap':0}
    path=BASE/'cohorts.json'
    if path.exists() and json.loads(path.read_text())!=data:
        raise ValueError('generated source cohorts changed')
    write(path,data);print('cohorts prepared',len(tr),len(va),data['n_train_keys'],data['n_validation_keys'],flush=True)


def neutral_generations(tok,model,rows,output):
    from cueconf.runner import generate_free
    signature=hashlib.sha256(json.dumps([asdict(x) for x in rows],sort_keys=True).encode()).hexdigest()
    if output.exists():
        previous=json.loads(output.read_text())
        if previous['complete'] and previous['source_signature']==signature:
            return previous['records']
        raise ValueError('generated source cache differs')
    ds=make_demos(3,'word',seed=7); records=[]
    for offset in range(0,len(rows),64):
        chunk=rows[offset:offset+64]
        gens=generate_free(tok,model,[build_prompt(x,ds,'cot') for x in chunk],128,8,stop_at_newline=True)
        if len(gens)!=len(chunk):raise ValueError('incomplete neutral generation')
        records.extend({'id':x.id,'generation':g,'demo_seed':7} for x,g in zip(chunk,gens))
        print(output.name,len(records),'/',len(rows),flush=True)
    write(output,{'complete':True,'source_signature':signature,'records':records,
                  'greedy':True,'max_new_tokens':128,'selection':'all source rows regardless of generated correctness'})
    return records


def extract(tok,model,rows,generations,role,seed,destination,layers):
    import torch
    ds=make_demos(3,'word',seed=seed)
    regime='cot' if seed==7 else f'cot_s{seed}'
    signature=hashlib.sha256(json.dumps({'rows':[asdict(x) for x in rows],
        'generations':generations,'role':role,'seed':seed,'layers':layers},sort_keys=True).encode()).hexdigest()
    meta_path=destination/'meta.json'; state_path=destination/'states.npy'
    if meta_path.exists() and state_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['complete'] and meta['signature']==signature:
            return meta
        raise ValueError('generated-state cache differs')
    pending=[]; records=[]
    by_id={r['id']:r for r in generations}
    for x in rows:
        generated=by_id[x.id]['generation']; boundary=value_boundary(generated,x.names[role])
        record={'id':x.id,'set_id':x.set_id,'program_key':computation_key(x),'name':x.names[role],
                'true_value':x.values[role],'lure':x.lure,'condition':x.condition,
                'written_value':boundary['value'] if boundary else None,'boundary_valid':False}
        for key in ('primary_sample','lure_augmentation'):
            if key in by_id[x.id]:record[key]=by_id[x.id][key]
        records.append(record)
        if not boundary:continue
        prompt=build_prompt(x,ds,regime)
        enc=tok(prompt+generated,return_offsets_mapping=True,add_special_tokens=True)
        marker=len(prompt)+boundary['marker']
        index=next((i for i,(a,b) in enumerate(enc['offset_mapping']) if a<=marker<b),None)
        if index is None:continue
        try:assert_before_value(enc['offset_mapping'],index,len(prompt)+boundary['value_start'])
        except ValueError:continue
        record['boundary_valid']=True
        record['state_index']=len(pending)
        pending.append(enc['input_ids'][:index+1])
    if not pending:raise ValueError('no valid generated value boundaries')
    destination.mkdir(parents=True,exist_ok=True)
    from cueconf.runner import text_config
    dim=text_config(model).hidden_size
    temporary=destination/'states.tmp.npy'
    states=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(pending),len(layers),dim))
    for offset in range(0,len(pending),8):
        chunk=pending[offset:offset+8]; T=max(map(len,chunk))
        ids=torch.full((len(chunk),T),tok.pad_token_id,device='cuda',dtype=torch.long); attention=torch.zeros_like(ids)
        for i,seq in enumerate(chunk):
            ids[i,:len(seq)]=torch.tensor(seq,device='cuda');attention[i,:len(seq)]=1
        with torch.no_grad():
            hs=model(input_ids=ids,attention_mask=attention,output_hidden_states=True).hidden_states
            h=torch.stack([torch.stack([hs[layer][i,len(seq)-1].float() for layer in layers]) for i,seq in enumerate(chunk)])
            if not torch.isfinite(h).all():raise ValueError('nonfinite generated hidden states')
            states[offset:offset+len(chunk)]=h.cpu().numpy()
        del hs,h
    states.flush();del states;temporary.replace(state_path)
    meta={'complete':True,'signature':signature,'role':role,'seed':seed,'layers':layers,
          'n_source':len(rows),'n_valid':len(pending),'records':records,'pre_value_boundary_checked':True,
          'storage_dtype':'float32','states_shape':[len(pending),len(layers),dim]}
    write(meta_path,meta);print('cached',destination,len(pending),'/',len(rows),flush=True)
    return meta


def train_and_score(root,role,layers,test_paths):
    import torch
    from cueconf.probes import BatchedProbes
    train_dir=root/'train'/role; val_dir=root/'validation'/role
    tr=json.loads((train_dir/'meta.json').read_text()); va=json.loads((val_dir/'meta.json').read_text())
    valid=lambda meta:[r for r in meta['records'] if r['boundary_valid']]
    train_rows=valid(tr); val_rows=valid(va)
    if len(train_rows)<500 or len(val_rows)<100:raise ValueError('too few generated training or validation boundaries')
    signature=hashlib.sha256(json.dumps({'train':tr['signature'],'validation':va['signature'],
                                       'epochs':10000,'lr':.001,'seeds':[0,1,2]},sort_keys=True).encode()).hexdigest()
    weights=root/f'{role}_weights.npz'; meta_file=root/f'{role}_selection.json'
    Htr=np.load(train_dir/'states.npy',mmap_mode='r'); Hval=np.load(val_dir/'states.npy',mmap_mode='r')
    d=Htr.shape[-1]
    if weights.exists() and meta_file.exists():
        selection=json.loads(meta_file.read_text())
        if selection['signature']!=signature:raise ValueError('generated probe weights differ')
        with np.load(weights) as stored:
            W,b,cW,cb=(torch.tensor(stored[k],device='cuda') for k in ('W','b','cW','cb'))
    else:
        X=torch.tensor(np.asarray(Htr),device='cuda').permute(1,0,2).contiguous()
        Xk=X.repeat_interleave(3,dim=0)
        y=torch.tensor([r['true_value'] for r in train_rows],device='cuda')
        ctl=lambda r:hashlib.sha256(f"0:{r['name']}".encode()).digest()[0]%10
        probes=BatchedProbes(len(layers)*3,d,[0,1,2],'cuda');probes.fit_sgd(Xk,y)
        controls=BatchedProbes(len(layers)*3,d,[0,1,2],'cuda')
        controls.fit_sgd(Xk,torch.tensor([ctl(r) for r in train_rows],device='cuda'))
        scores=[]; vy=np.array([r['true_value'] for r in val_rows]); vc=np.array([ctl(r) for r in val_rows])
        for index,layer in enumerate(layers):
            v=torch.tensor(np.asarray(Hval[:,index]),device='cuda')
            predictions=torch.stack([probes.probe(index*3+s,d).logprobs(v).argmax(-1) for s in range(3)]).cpu().numpy()
            cp=torch.stack([controls.probe(index*3+s,d).logprobs(v).argmax(-1) for s in range(3)]).cpu().numpy()
            scores.append({'layer':layer,'accuracy':float(np.mean(predictions==vy)),
                           'control_accuracy':float(np.mean(cp==vc))})
        best=max(range(len(scores)),key=lambda i:scores[i]['accuracy'])
        W=probes.W[best*3:best*3+3].clone();b=probes.b[best*3:best*3+3].clone()
        cW=controls.W[best*3:best*3+3].clone();cb=controls.b[best*3:best*3+3].clone()
        np.savez_compressed(weights,W=W.cpu().numpy(),b=b.cpu().numpy(),cW=cW.cpu().numpy(),cb=cb.cpu().numpy())
        selection={'signature':signature,'role':role,'layer':layers[best],'validation_scores':scores,
                   'n_train_valid':len(train_rows),'n_validation_valid':len(val_rows),
                   'train_signature':tr['signature'],'validation_signature':va['signature'],
                   'rule':'highest mean validation accuracy; lowest-layer tie break'}
        write(meta_file,selection)
        del X,Xk,probes,controls;torch.cuda.empty_cache()
    li=layers.index(selection['layer'])
    for path in test_paths:
        destination=path/'readouts.json'; test=json.loads((path/'meta.json').read_text())
        signature_test=hashlib.sha256((test['signature']+signature).encode()).hexdigest()
        if destination.exists():
            previous=json.loads(destination.read_text())
            if previous['complete'] and previous['signature']==signature_test:continue
            raise ValueError('test readouts changed')
        H=np.load(path/'states.npy',mmap_mode='r')
        rows=[]
        for record in test['records']:
            record=dict(record)
            if record['boundary_valid']:
                h=torch.tensor(np.asarray(H[record['state_index'],li]),device='cuda')
                with torch.no_grad():
                    lp=torch.log_softmax(torch.einsum('d,kcd->kc',h,W)+b,dim=-1)
                    cp=(torch.einsum('d,kcd->kc',h,cW)+cb).argmax(-1)
                    if not torch.isfinite(lp).all():raise ValueError('nonfinite generated test readout')
                    pred=lp.argmax(-1).cpu().numpy()
                    record['predictions']=pred.tolist()
                    record['probe_accuracy']=float(np.mean(pred==record['true_value']))
                    ctl=hashlib.sha256(f"0:{record['name']}".encode()).digest()[0]%10
                    record['control_accuracy']=float(np.mean(cp.cpu().numpy()==ctl))
                    if record['lure'] is not None:
                        record['margin']=float((lp[:,record['true_value']]-lp[:,record['lure']]).mean())
            rows.append(record)
        write(destination,{'complete':True,'signature':signature_test,'model':root.name,'role':role,
                           'seed':test['seed'],'layer':selection['layer'],'records':rows,
                           'test_signature':test['signature'],'weight_signature':signature})
        print('scored',destination,flush=True)


def run(model_key):
    import torch
    from cueconf.runner import load_model,text_config
    source=json.loads((BASE/'cohorts.json').read_text())
    cohorts={key:[Instance(**r) for r in source[key]] for key in ('train','validation')}
    root=BASE/model_key;root.mkdir(parents=True,exist_ok=True)
    tok,model=load_model(model_entry(model_key)['hf_id'])
    nl=text_config(model).num_hidden_layers;layers=sorted(set(range(0,nl+1,2))|{nl})
    for split,rows in cohorts.items():
        generations=neutral_generations(tok,model,rows,root/f'{split}_generations.json')
        for role in ('v1','v2'):extract(tok,model,rows,generations,role,7,root/split/role,layers)
    all_instances={x.id:x for x in read_jsonl(DATA_DIR/'L3/test_sets.jsonl')}
    paths={role:[] for role in ('v1','v2')}
    for role in ('v1','v2'):
        for seed in (7,11,13):
            regime='cot' if seed==7 else f'cot_s{seed}'
            old=json.loads((OUT_DIR/'round5_generated'/model_key/regime/f'{role}.json').read_text())
            requested={r['id']:r for r in old['records']}
            behaviors={}
            for condition in ('neutral',f'incongruent@{role}'):
                p=OUT_DIR/'runs'/model_key/'L3'/regime/condition/'behavior.jsonl'
                behaviors.update({r['id']:r for r in (json.loads(line) for line in p.open())})
            rows=[all_instances[id_] for id_ in requested]
            generations=[{**behaviors[x.id],**{k:requested[x.id][k] for k in ('primary_sample','lure_augmentation')}} for x in rows]
            path=root/'test'/f's{seed}'/role
            extract(tok,model,rows,generations,role,seed,path,layers);paths[role].append(path)
    del model;torch.cuda.empty_cache()
    for role in ('v1','v2'):train_and_score(root,role,layers,paths[role])
    write(root/'complete.json',{'complete':True,'model':model_key,'roles':['v1','v2'],'seeds':[7,11,13]})


def smoke(model_key):
    import torch
    from cueconf.runner import load_model,text_config
    from cueconf.probes import BatchedProbes
    source=json.loads((BASE/'cohorts.json').read_text())
    rows=[Instance(**r) for r in source['validation'][:16]]
    tok,model=load_model(model_entry(model_key)['hf_id'])
    layers=[0,text_config(model).num_hidden_layers]
    root=BASE/'smoke'/model_key
    generated=neutral_generations(tok,model,rows,root/'generations.json')
    for role in ('v1','v2'):
        path=root/role
        meta=extract(tok,model,rows,generated,role,7,path,layers)
        H=np.load(path/'states.npy')
        X=torch.tensor(H[:,-1],device='cuda').unsqueeze(0).repeat(3,1,1)
        labels=torch.tensor([r['true_value'] for r in meta['records'] if r['boundary_valid']],device='cuda')
        probe=BatchedProbes(3,X.shape[-1],[0,1,2],'cuda');probe.fit_sgd(X,labels,epochs=5)
        if not torch.isfinite(probe.W).all():raise ValueError('smoke probe weights nonfinite')
    write(root/'passed.json',{'passed':True,'model':model_key,'source_rows':16,'roles':['v1','v2'],
                              'check':'actual generated prefixes, float32 hidden states and five SGD steps'})
    print('GPU smoke passed',model_key,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--model')
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.prepare:prepare()
    elif a.model and a.smoke:smoke(a.model)
    elif a.model:run(a.model)
    else:ap.error('use --prepare or --model')


if __name__=='__main__':main()
