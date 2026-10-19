const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const dir=path.join(__dirname,'build','column-v2-export');
(async()=>{
  const reports=[];
  for(const variant of ['scalar','simd']){
    const e=await require(`./build/engine-${variant}.js`)({print:()=>{},printErr:console.error});
    const call=(name,type='number',types=[],args=[])=>e.ccall(name,type,types,args);
    assert.equal(call('web_init','number',['number','number','number'],[24,24,20261018]),1);
    assert.equal(JSON.parse(call('web_status','string',['number'],[1])).protocol,'column-v2');
    let state=20261024;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state;};
    for(const precision of ['','-fp8','-fp4']){
      const file='column-v2-final'+precision+'.cn';e.FS.writeFile('/model.cn',fs.readFileSync(path.join(dir,file)));
      assert.equal(call('web_load','number',['string','number'],['/model.cn',0]),1);
      assert.equal(call('web_protocol','number',['number','number'],[0,2]),1);
      let transitions=0;
      for(let a=0;a<10;a++)for(let b=0;b<10;b++)for(let ci=0;ci<2;ci++){
        /* Prefix a carry-generating lower column to test ci=1 through inference. */
        const aa=ci?String(a)+'9':String(a),bb=ci?String(b)+'1':String(b);
        assert.equal(call('web_predict','number',['string','string','number','number'],[aa,bb,0,1]),1);
        const r=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'})),col=r.columns[ci?1:0];
        assert.ok(r.valid);assert.equal(col.a,a);assert.equal(col.b,b);assert.equal(col.cin,ci);
        assert.equal(col.digit,(a+b+ci)%10);assert.equal(col.cout,Math.floor((a+b+ci)/10));
        assert.equal(col.ids.length,5);assert.deepEqual(col.ids.slice(3),[257,258]);
        assert.ok(col.input_ids.includes(117));assert.ok(!col.input_ids.includes(259));transitions++;
      }
      const widths=[];
      for(const width of [1,2,4,6,10]){
        for(let k=0;k<40;k++){
          const operand=()=>Array.from({length:width},()=>String(random()%10)).join('');
          const a=k===0?'9'.repeat(width):operand(),b=k===0?'1':operand();
          assert.equal(call('web_predict','number',['string','string','number','number'],[a,b,0,1]),1);
          const r=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));assert.ok(r.valid);assert.equal(BigInt(r.answer),BigInt(a)+BigInt(b));
          const raw=e.FS.readFile('/inference.txt',{encoding:'utf8'});assert.equal((raw.match(/<think>/g)||[]).length,1);assert.equal((raw.match(/<\/think>/g)||[]).length,1);
          assert.equal((raw.match(/<\|im_start\|>user/g)||[]).length,r.columns.length);
        }
        widths.push({width,correct:40,total:40});
      }
      assert.equal(call('web_predict','number',['string','string','number','number'],['9'.repeat(1000),'1',0,1]),1);
      const long=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));assert.equal(long.answer,'1'+'0'.repeat(1000));
      reports.push({variant,file,transitions,widths,carry1000:true});console.log(`${variant} ${file}: 200 transitions, 200 whole problems and 1000-digit carry passed`);
    }
    assert.equal(call('web_protocol','number',['number','number'],[1,2]),1);
    for(let stage=1;stage<=6;stage++){assert.equal(call('web_begin','number',['number','number','number','number'],[stage,1,1,0]),1);assert.equal(call('web_tick'),1);}
  }
  fs.writeFileSync(path.join(__dirname,'build','validation-column-wasm.json'),JSON.stringify({passed:true,reports},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
