/* Validate real whole-question generation on both delivered WASM engines. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const version=process.argv.includes('--v4')?'v4':process.argv.includes('--v5')?'v5':'v6',prefix='dialogue-'+version;
const out=path.join(__dirname,'build'),exportsDir=path.join(out,prefix+'-export');
const native=JSON.parse(fs.readFileSync(path.join(out,prefix+'-tests/report.json')));
const special={256:'<|im_start|>',257:'<|im_end|>',258:'<|endoftext|>',259:'<think>',260:'</think>'};
const decode=t=>t.map(n=>special[n]??(n===10||n>=32&&n<127?String.fromCharCode(n):`\\x${n.toString(16).padStart(2,'0')}`)).join('');
(async()=>{
 const reports=[];
 for(const variant of ['scalar','simd']){
  const e=await require(`./build/engine-${variant}.js`)({print:()=>{},printErr:console.error});
  const call=(n,t='number',ts=[],args=[])=>e.ccall(n,t,ts,args);
  assert.equal(call('web_init','number',['number','number','number'],[24,24,20261018]),1);
  const status=()=>JSON.parse(call('web_status','string',['number'],[1]));
  const reward = text => call('web_outcome_reward','number',['string','string','number','string'],['9','1',1,text]);
  assert.equal(reward('<think>9+1=999; revise freely</think>10<|im_end|><|endoftext|>'),1);
  assert.equal(reward('<think></think>10<|im_end|><|endoftext|>'),1);
  assert.equal(reward('<think>/think</think>10<|im_end|><|endoftext|>'),0);
  assert.equal(reward('<think>draft</think>11<|im_end|><|endoftext|>'),0);
  assert.equal(call('web_shape','number',['number','number','number'],[64,128,6]),0);
  for(const [layers,dim,ff,count] of [[6,56,128,462412],[10,40,104,448788]]){
    assert.equal(call('web_init_arch','number',['number','number','number','number'],[dim,ff,layers,20261018]),1);
    assert.equal(status().parameters,count);assert.equal(status().layers,layers);
  }
  assert.equal(call('web_init','number',['number','number','number'],[24,24,20261018]),1);

  assert.equal(status().protocol,'chat-v2');assert.equal(status().window,512);
  const decodeExample=(format,cot=1)=>{
   const p=JSON.parse(call('web_example_format','string',['string','string','number','number','number'],['99','1',cot,0,format]));
   assert.equal(p.tokens.filter(t=>t===259).length,cot);
   assert.equal(p.tokens.filter(t=>t===260).length,cot);
   assert.ok(!p.tokens.slice(0,p.prompt).some(t=>t===259||t===260));
   return decode(p.tokens);
  };
  assert.equal(decodeExample(1),'<|im_start|>user\n99+1<|im_end|>\n<|im_start|>assistant\n<think>\npos=0 a=9 b=1 cin=0 -> 0 1\npos=1 a=9 b=0 cin=1 -> 0 1\n</think>\n100<|im_end|><|endoftext|>');
  assert.ok(decodeExample(0).includes('9+1+0=0,1\n9+0+1=0,1'));
  assert.ok(!decodeExample(1,0).includes('<think>'));
  /* Historical metadata keeps its original tokenizer targets and context.
     Loading a compact checkpoint never silently opts it into the new CoT. */
  e.FS.writeFile('/compact.cn',fs.readFileSync(path.join(out,'dialogue-v3-export/dialogue-v3-stage-6.cn')));
  assert.equal(call('web_load','number',['string','number'],['/compact.cn',1]),1);
  assert.equal(call('web_protocol','number',['number','number'],[1,1]),1);
  assert.equal(call('web_protocol','number',['number','number'],[1,3]),0);
  assert.equal(status().protocol,'chat-v1');assert.equal(status().window,256);
  assert.equal(call('web_predict','number',['string','string','number','number'],['99','1',1,1]),1);
  const old=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));
  const oldNative=JSON.parse(fs.readFileSync(path.join(out,'dialogue-v3-tests/short.json')));
  assert.deepEqual(old.tokens,oldNative.tokens);assert.equal(old.protocol,'chat-v1');
  const stages=[];
  for(let stage=1;stage<=(version==='v6'?7:6);stage++){
   e.FS.writeFile('/input.cn',fs.readFileSync(path.join(exportsDir,`${prefix}-stage-${stage}.cn`)));
   assert.equal(call('web_load','number',['string','number'],['/input.cn',0]),1);
   assert.equal(call('web_protocol','number',['number','number'],[0,3]),1);
   assert.equal(call('web_predict','number',['string','string','number','number'],['99','1',0,1]),1);
   const p=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));
   const raw=e.FS.readFile('/inference.txt',{encoding:'utf8'});
   assert.equal(p.protocol,'chat-v2');assert.deepEqual(p.columns,[]);
   assert.equal(raw,decode(p.input_ids)+decode(p.tokens));
   stages.push({stage,valid:p.valid,answer:p.answer,generated:p.generated,stop:p.stop});
  }
  const samples=[];
  for(const name of ['complaint','short','fields','carry10','rollover']){
   const expected=native.results.find(r=>r.name===name);
   assert.equal(call('web_predict','number',['string','string','number','number'],[expected.a,expected.b,0,1]),1);
   const p=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));
   const raw=e.FS.readFile('/inference.txt',{encoding:'utf8'});
   const tokensEqual=JSON.stringify(p.tokens)===JSON.stringify(expected.tokens);
   assert.deepEqual(p.input_ids,expected.input_ids);assert.equal(raw,decode(p.input_ids)+decode(p.tokens));
   samples.push({name,generated:p.generated,valid:p.valid,answer:p.answer,correct:p.valid&&p.answer.replace(/^0+(?=\d)/,'')===expected.expected,rollovers:p.rollovers,
                 native_tokens_equal:tokensEqual,...(!tokensEqual?{native_tokens:expected.tokens,wasm_tokens:p.tokens,raw}:{} )});
  }
  const scores={};
  for(const expected of native.results.filter(r=>/^w\d+-/.test(r.name))){
   assert.equal(call('web_predict','number',['string','string','number','number'],[expected.a,expected.b,0,1]),1);
   const p=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));
   const score=scores[expected.width]??={correct:0,total:0,native_tokens_equal:0};
   score.total++;score.correct+=p.valid&&p.answer.replace(/^0+(?=\d)/,'')===expected.expected;score.native_tokens_equal+=JSON.stringify(p.tokens)===JSON.stringify(expected.tokens);
   if(JSON.stringify(p.tokens)!==JSON.stringify(expected.tokens)){
    const dir=path.join(out,version==='v6'?'outcome-v6':'capacity-v5',`wasm-${variant}`);fs.mkdirSync(dir,{recursive:true});
    fs.writeFileSync(path.join(dir,expected.name+'.json'),JSON.stringify(p,null,2));
    fs.writeFileSync(path.join(dir,expected.name+'.txt'),e.FS.readFile('/inference.txt'));
   }
  }
  // Short actual updates in CoT and direct modes, including failed checkpoints.
  const updates=[];
  for(const cot of [0,1]){
   assert.equal(call('web_init','number',['number','number','number'],version==='v5'?[48,96,19]:[24,24,19]),1);
   assert.equal(status().parameters,version==='v5'?314812:57388);
   if(version==='v5'){assert.equal(status().headDim,16);assert.equal(status().qRank,16);assert.equal(status().engramDim,4);}
   assert.equal(call('web_mode','number',['number'],[cot]),1);
   for(let stage=1;stage<=(version==='v6'?7:6);stage++){
    assert.equal(call('web_begin','number',['number','number','number','number'],[stage,1,1,0]),1);
    assert.equal(call('web_tick'),1);assert.equal(status().protocol,'chat-v2');
    for(const suffix of ['-entry','-last',''])assert.ok(e.FS.readFile(`/experiment/stage-${stage}${suffix}.cn`).length);
    if(stage===7){assert.equal(call('web_rollouts'),1);const t=JSON.parse(e.FS.readFile('/rollouts.json',{encoding:'utf8'}));assert.equal(t.samples.length,4);for(const p of t.samples) assert.equal(call('web_outcome_reward','number',['string','string','number','string'],[t.a,t.b,cot,decode(p.tokens.slice(t.prompt))]),p.reward);}
    updates.push({cot,stage,step:status().step,correct:status().correct});
   }
  }
  reports.push({variant,stages,samples,updates,scores});
  console.log(`${variant}: protocol/training passed; native tokens ${samples.filter(s=>s.native_tokens_equal).length}/${samples.length} representative cases; scores ${JSON.stringify(scores)}`);
 }
 const generatedEquivalent=reports.every(r=>r.samples.every(s=>s.native_tokens_equal)&&Object.values(r.scores).every(s=>s.native_tokens_equal===s.total));
 fs.writeFileSync(path.join(out,version==='v6'?'validation-outcome-wasm.json':version==='v5'?'validation-capacity-wasm.json':'validation-fields-wasm.json'),JSON.stringify({passed:generatedEquivalent,protocol_passed:true,generation_equivalence_passed:generatedEquivalent,scope:'Protocol and real training checks passed; generated-token disagreements retained, arithmetic scores measured independently.',reports},null,2));
 if(!generatedEquivalent)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
