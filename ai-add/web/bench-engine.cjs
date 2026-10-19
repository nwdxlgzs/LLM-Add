/* Development benchmark: fixed prompt, warmed engine, median timings, no DOM. */
const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
(async () => {
  const create = require('./build/engine-simd.js');
  const e = await create({ print() {}, printErr: console.error });
  const call = (name, type, types, args) => e.ccall(name, type, types, args);
  call('web_init', 'number', ['number','number','number'], [24,24,20261018]);
  const rows = [];
  for (const file of ['final.cn','final-fp8.cn','final-fp4.cn']) {
    e.FS.writeFile('/bench.cn', fs.readFileSync(path.join(__dirname,'../run-final',file)));
    call('web_load','number',['string','number'],['/bench.cn',0]);
    for (const cache of e._web_cache ? [0,1] : [0]) {
      if (e._web_cache) call('web_cache','number',['number','number'],[0,cache]);
      const run = () => call('web_predict','number',['string','string','number','number'],['12345678901234567890','98765432109876543210',0,1]);
      for (let i=0;i<4;i++) run();
      const times=[];
      for (let i=0;i<15;i++) { const start=performance.now(); if(!run())throw Error('inference failed'); times.push(performance.now()-start); }
      const sorted=[...times].sort((a,b)=>a-b);
      rows.push({file,cache:!!cache,median_ms:sorted[7],times_ms:times});
    }
  }
  const report={backend:'wasm-simd128',protocol:'legacy-column',input_digits:20,warmups:4,repetitions:15,rows};
  const name=path.basename(process.argv[2]||'benchmark-quant.json');
  fs.writeFileSync(path.join(__dirname,'build',name),JSON.stringify(report,null,2));
  console.log(JSON.stringify(rows.map(({times_ms,...r})=>r),null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
