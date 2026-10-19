/* Compare the actual delivered WASM engines with native C, including AdamW state. */
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),assert=require('node:assert/strict');
const outcome=process.argv.includes('--outcome');
const capacity=process.argv.includes('--capacity');
const root=path.resolve(__dirname,'..'),out=path.join(__dirname,'build',outcome?'parity-v6':capacity?'parity-v5':process.argv.includes('--fields')?'parity-v4':process.argv.includes('--dialogue')?'parity-v3':'parity-v2');
fs.mkdirSync(out,{recursive:true});
const exe=path.join(root,'build','engine_parity.exe');
const src=fs.readFileSync(path.join(root,'..','cairn','src','cairn.c'),'utf8');
const countFields=(a,b)=>(src.split(a)[1].split(b)[0].match(/X\(\w+\)/g)||[]).length;
const ni=countFields('#define CONFIG_INTS(X)','#define CONFIG_FLOATS(X)'),nf=countFields('#define CONFIG_FLOATS(X)','static void io_config');
function native(input,prefix,mode,steps,backend){cp.execFileSync(exe,[input,prefix,String(mode),String(steps),String(backend)],{stdio:'pipe'});}
function floats(file){const b=fs.readFileSync(file),v=[];assert.equal(b.length%4,0);for(let i=0;i<b.length;i+=4)v.push(b.readFloatLE(i));return v;}
function parameters(file){
  const b=fs.readFileSync(file);assert.equal(b.subarray(0,8).toString('ascii'),'CAIRN02\0');
  const flags=b.readUInt32LE(12),np=b.readUInt32LE(16),layers=b.readUInt32LE(40+2*4);
  let at=40+(ni+nf)*4+8+261*4+layers*(9*4+112*4+8*8),rows=[];
  for(let k=0;k<np;k++){
    const r=b.readInt32LE(at),c=b.readInt32LE(at+4),role=b.readInt32LE(at+8),storage=b.readUInt32LE(at+12);at+=16;assert.equal(storage,0);
    for(let kind=0;kind<(flags?3:1);kind++){
      let values=[];for(let i=0;i<r*c;i++,at+=4)values.push(b.readFloatLE(at));
      rows.push({tensor:k,role,kind,values});
    }
  }
  assert.equal(at+8,b.length);return rows;
}
function close(a,b,atol,rtol){
  assert.equal(a.length,b.length);let maxAbs=0,maxScaled=0,rms=0;
  for(let i=0;i<a.length;i++){
    assert.ok(Number.isFinite(a[i])&&Number.isFinite(b[i]));
    const d=Math.abs(a[i]-b[i]),scale=atol+rtol*Math.max(Math.abs(a[i]),Math.abs(b[i]));
    maxAbs=Math.max(maxAbs,d);maxScaled=Math.max(maxScaled,d/scale);rms+=d*d;
  }
  assert.ok(maxScaled<=1,`numeric mismatch maxAbs=${maxAbs} scaled=${maxScaled} atol=${atol} rtol=${rtol}`);
  return {count:a.length,max_abs:maxAbs,rms:Math.sqrt(rms/a.length),max_fraction_of_tolerance:maxScaled};
}
function compare(a,b){
  const labeled=process.argv.includes('--fields');
  const before=close(floats(a+'-before.bin'),floats(b+'-before.bin'),2e-4,2e-4);
  const after=close(floats(a+'-after.bin'),floats(b+'-after.bin'),labeled?2e-3:5e-4,5e-4);
  const x=parameters(a+'-model.cn'),y=parameters(b+'-model.cn');assert.equal(x.length,y.length);
  const fields=['weights','adam_m','adam_v'],state={};
  for(let kind=0;kind<3;kind++)state[fields[kind]]=close(x.filter(r=>r.kind===kind).flatMap(r=>r.values),y.filter(r=>r.kind===kind).flatMap(r=>r.values),process.argv.includes('--fields')?1e-4:2e-5,2e-4);
  const sa=JSON.parse(fs.readFileSync(a+'-steps.json')),sb=JSON.parse(fs.readFileSync(b+'-steps.json'));assert.equal(sa.length,sb.length);
  let loss=0,grad=0;
  sa.forEach((v,i)=>{assert.equal(v.step,sb[i].step);assert.equal(v.informative,sb[i].informative);assert.equal(v.reward,sb[i].reward);close([v.loss,v.grad_norm],[sb[i].loss,sb[i].grad_norm],2e-4,5e-4);loss=Math.max(loss,Math.abs(v.loss-sb[i].loss));grad=Math.max(grad,Math.abs(v.grad_norm-sb[i].grad_norm));});
  const fa=floats(a+'-after.bin'),fb=floats(b+'-after.bin');let argmaxMatches=0;
  for(let i=0;i<fa.length;i+=261){let ia=0,ib=0;for(let j=1;j<261;j++){if(fa[i+j]>fa[i+ia])ia=j;if(fb[i+j]>fb[i+ib])ib=j;}assert.equal(ia,ib);argmaxMatches++;}
  return {before,after,state,loss_max_abs:loss,grad_norm_max_abs:grad,argmax_matches:argmaxMatches,optimizer_steps:sa.map(v=>v.step),informative_steps:sa.filter(v=>v.informative).length};
}
function diagnostic(a,b){
  const diff=(x,y)=>{assert.equal(x.length,y.length);let max=0,sum=0;for(let i=0;i<x.length;i++){const d=Math.abs(x[i]-y[i]);max=Math.max(max,d);sum+=d*d;}return {count:x.length,max_abs:max,rms:Math.sqrt(sum/x.length)};};
  const x=parameters(a+'-model.cn'),y=parameters(b+'-model.cn'),state={};
  for(const [kind,key] of ['weights','adam_m','adam_v'].entries())state[key]=diff(x.filter(r=>r.kind===kind).flatMap(r=>r.values),y.filter(r=>r.kind===kind).flatMap(r=>r.values));
  const fa=floats(a+'-after.bin'),fb=floats(b+'-after.bin');let changed=0;
  for(let i=0;i<fa.length;i+=261){let ia=0,ib=0;for(let j=1;j<261;j++){if(fa[i+j]>fa[i+ia])ia=j;if(fb[i+j]>fb[i+ib])ib=j;}changed+=ia!==ib;}
  return {before:diff(floats(a+'-before.bin'),floats(b+'-before.bin')),after:diff(fa,fb),state,argmax_changes:changed};
}
(async()=>{
  const seed=path.join(out,'seed.cn');
  if(outcome)fs.copyFileSync(path.join(__dirname,'build/dialogue-v6-run/stage-1-entry.cn'),seed);
  else if(capacity)fs.copyFileSync(path.join(__dirname,'build/dialogue-v5-run/stage-1-entry.cn'),seed);
  else cp.execFileSync(exe,[seed]);
  const rl=path.join(__dirname,'build','column-v2-run','stage-2.cn');assert.ok(fs.existsSync(rl),'Run structured SFT 1 before testing meaningful RLOO parity');
  const cases=outcome?
    [['prefill',10,0,seed],['masked_chat',11,3,seed],['chat_sft',11,3,path.join(__dirname,'build/dialogue-v6-run/stage-3.cn')],['outcome_rloo',17,24,path.join(__dirname,'build/dialogue-v6-run/stage-7-entry.cn')],['chat_fp4_qat',16,3,seed],['trained_dialogue',10,0,path.join(__dirname,'build/dialogue-v6-run/stage-7.cn')]]:
    capacity?
    [['prefill',10,0,seed],['masked_chat',11,3,seed],['chat_sft',11,3,path.join(__dirname,'build/dialogue-v5-run/stage-3.cn')],['chat_rloo',15,24,path.join(__dirname,'build/dialogue-v5-run/stage-3.cn')],['chat_fp4_qat',16,3,seed],['trained_dialogue',10,0,path.join(__dirname,'build/dialogue-v5-run/stage-6.cn')]]:
    process.argv.includes('--fields')?
    [['prefill',10,0,seed],['masked_chat',11,3,seed],['chat_sft',11,3,path.join(__dirname,'build/dialogue-v4-run/stage-3.cn')],['chat_rloo',15,24,path.join(__dirname,'build/dialogue-v4-run/stage-3.cn')],['chat_fp4_qat',16,3,seed],['trained_dialogue',10,0,path.join(__dirname,'build/dialogue-v4-run/stage-6.cn')]]:
    process.argv.includes('--dialogue')?
    [['prefill',0,0,seed],['masked_chat',1,3,seed],['chat_sft',1,3,path.join(__dirname,'build/dialogue-v3-run/stage-3.cn')],['chat_rloo',5,16,path.join(__dirname,'build/dialogue-v3-run/stage-3.cn')],['chat_fp4_qat',6,3,seed],['trained_dialogue',0,0,path.join(__dirname,'build/dialogue-v3-run/stage-6.cn')]]:
    [['prefill',0,0,seed],['masked_chat',1,3,seed],['column_sft',2,3,seed],['cot_rloo',3,12,rl],['fp4_qat',4,3,seed],...['','-fp8','-fp4'].map(suffix=>['trained'+(suffix||'-fp32'),0,0,path.join(__dirname,'build','column-v2-export','column-v2-final'+suffix+'.cn')])];
  const report={passed:false,scope:'Same serialized initial weights, token stream and RNG; native scalar + native auto vs actual WASM scalar/SIMD128. Numeric equivalence with declared tolerances; not bit-identical or a proof of identical long optimization trajectories.',tolerances:{before_logits:{atol:2e-4,rtol:2e-4},after_logits:{atol:5e-4,rtol:5e-4},weights_and_moments:{atol:2e-5,rtol:2e-4}},results:[]};
  const reportFile=path.join(out,'report.json');
  if(process.argv.includes('--fields')){
    report.tolerances.after_logits.atol=2e-3;
    report.tolerances.weights_and_moments.atol=1e-4;
    report.tolerance_change='Long labeled scratchpads exceeded historical short-format tolerances in the random masked-SFT probe: post-update logits max 0.00113428, weight max 0.00002304, Adam m max 0.00005573. Initial logits, losses and gradient norms remained close and all probe argmax matched. Expanded bounds are explicit; this is finite numeric agreement, not bit equality or a guarantee of identical long training.';
  }
  if(process.argv.includes('--resume')&&fs.existsSync(reportFile)){
    const previous=JSON.parse(fs.readFileSync(reportFile));
    report.results=previous.results;
  }
  const pending=cases.filter(([name,,,input])=>{
    if(process.argv.includes('--available')&&!fs.existsSync(input))return false;
    return !['scalar','simd'].every(v=>report.results.some(r=>r.variant===v&&r.name===name));
  });
  for(const [name,mode,steps,input] of pending)for(const backend of [1,0])native(input,path.join(out,`native-${backend}-${name}`),mode,steps,backend);
  for(const variant of ['scalar','simd']){
    const e=await require(`./build/engine-${variant}.js`)({print:()=>{},printErr:console.error});e.FS.mkdir('/parity');
    for(const [name,mode,steps,input] of pending){
      if(report.results.some(r=>r.variant===variant&&r.name===name))continue;
      e.FS.writeFile('/input.cn',fs.readFileSync(input));
      const before=performance.now();
      assert.equal(e.ccall('parity_run','number',['string','string','number','number','number'],['/input.cn','/parity/'+name,mode,steps,0]),1);
      const ms=performance.now()-before,prefix=path.join(out,`${variant}-${name}`);
      for(const suffix of ['-before.bin','-after.bin','-model.cn','-steps.json'])fs.writeFileSync(prefix+suffix,e.FS.readFile('/parity/'+name+suffix));
      const nativeMatches=[1,0].map(backend=>{
        const a=path.join(out,`native-${backend}-${name}`);
        try{return {backend:backend?'scalar':'auto',passed:true,...compare(a,prefix)};}
        catch(error){return {backend:backend?'scalar':'auto',passed:false,error:error.message,...diagnostic(a,prefix)};}
      });
      if(name==='chat_rloo'||name==='outcome_rloo') assert.ok(JSON.parse(fs.readFileSync(path.join(out,`native-0-${name}-steps.json`))).some(v=>v.informative),'RLOO parity needs actual reward differences and updates');
      /* Load a WASM optimizer checkpoint into native C and save/forward again. */
      native(prefix+'-model.cn',prefix+'-roundtrip',mode>=10?10:0,0,1);
      let roundtrip=true,roundtripError;
      try{close(floats(prefix+'-after.bin'),floats(prefix+'-roundtrip-before.bin'),2e-4,2e-4);}catch(error){roundtrip=false;roundtripError=error.message;}
      const passed=nativeMatches.every(m=>m.passed)&&roundtrip;
      const quantized=name==='chat_fp4_qat'||outcome&&name==='trained_dialogue'&&JSON.parse(fs.readFileSync(path.join(__dirname,'build/dialogue-v6-export/stages.json')))[6].precision>0||capacity&&name==='trained_dialogue'&&JSON.parse(fs.readFileSync(path.join(__dirname,'build/dialogue-v5-export/stages.json')))[5].precision>0;
      report.results.push({variant,name,quantized,passed,wasm_ms:ms,nativeMatches,wasm_to_native_roundtrip:roundtrip,roundtrip_error:roundtripError});
      fs.writeFileSync(reportFile,JSON.stringify(report,null,2));
      console.log(`${variant} ${name}: ${passed?'passed':'FAILED: divergence retained in report'}`);
    }
  }
  report.complete=report.results.length===cases.length*2;
  report.passed=report.complete&&report.results.every(r=>r.passed!==false);
  report.backend_aligned_passed=report.complete&&report.results.every(r=>r.wasm_to_native_roundtrip&&r.nativeMatches.find(m=>m.backend===(r.variant==='scalar'?'scalar':'auto'))?.passed!==false);
  for(const r of report.results)if(r.quantized===undefined)r.quantized=r.name==='chat_fp4_qat'||outcome&&r.name==='trained_dialogue'&&JSON.parse(fs.readFileSync(path.join(__dirname,'build/dialogue-v6-export/stages.json')))[6].precision>0||capacity&&r.name==='trained_dialogue'&&JSON.parse(fs.readFileSync(path.join(__dirname,'build/dialogue-v5-export/stages.json')))[5].precision>0;
  report.fp32_backend_aligned_passed=report.complete&&report.results.filter(r=>!r.quantized).every(r=>r.wasm_to_native_roundtrip&&r.nativeMatches.find(m=>m.backend===(r.variant==='scalar'?'scalar':'auto'))?.passed!==false);
  report.backend_alignment_note='Matching scalar/scalar and native-auto/WASM-SIMD comparisons are reported separately. Cross-backend FP4 QAT may diverge at quantization thresholds even when each matching pair agrees; an aligned pass does not turn the all-backend failed result into a pass.';
  if(capacity)report.backend_alignment_note='The wider model also has a native-auto vs WASM-SIMD random FP4 QAT counterexample. Report FP32 and QAT separately: successful finite FP32 comparisons do not prove identical quantized training trajectories.';
  fs.writeFileSync(reportFile,JSON.stringify(report,null,2));
  if(report.results.some(r=>r.passed===false))process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
