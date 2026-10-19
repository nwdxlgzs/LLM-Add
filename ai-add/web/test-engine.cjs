/* Node is only a development test runner; the delivered HTML needs no Node. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

async function check(variant, full) {
  const createEngine = require(`./build/engine-${variant}.js`);
  const e = await createEngine({ print: () => {}, printErr: console.error });
  const call = (name, type = 'number', types = [], args = []) => e.ccall(name, type, types, args);
  const status = () => JSON.parse(call('web_status', 'string', ['number'], [1]));
  assert.equal(call('web_init', 'number', ['number', 'number', 'number'], [24, 24, 20261018]), 1);
  assert.equal(status().parameters, 57388);
  assert.equal(status().backend, variant === 'simd' ? 'wasm-simd128' : 'scalar');
  assert.equal(status().protocol,'chat-v2');
  assert.equal(call('web_protocol','number',['number','number'],[1,1]),1);
  assert.equal(status().window,512);
  assert.equal(call('web_protocol','number',['number','number'],[1,0]),1);
  const certs = {};
  for (const file of ['final.cn', 'final-fp8.cn', 'final-fp4.cn']) {
    e.FS.writeFile('/loaded.cn', fs.readFileSync(path.join(root, 'run-final', file)));
    assert.equal(call('web_load', 'number', ['string', 'number'], ['/loaded.cn', 0]), 1);
    assert.equal(call('web_certify', 'number', ['number'], [0]), 1);
    certs[file] = JSON.parse(e.FS.readFile('/certificate.json', { encoding: 'utf8' }));
    for (const [a, b, answer] of [['9999999999', '9999999999', '19999999998'], ['9999999999', '1', '10000000000'], ['0000000000', '0', '0'], ['1234567890', '9876543210', '11111111100']]) {
      const result = JSON.parse(call('web_infer', 'string', ['string', 'string', 'number'], [a, b, 0]));
      assert.equal(result.ok, true); assert.equal(result.answer, answer);
      const raw = e.FS.readFile('/inference.txt', { encoding: 'utf8' });
      assert.ok(raw.startsWith(`<|im_start|>user\n${a}+${b}\n<|im_end|>\n<|im_start|>assistant\n<think>\n`));
      assert.ok(raw.endsWith(`</think>\n${answer}<|im_end|><|endoftext|>\n`));
      assert.equal((raw.match(/^pos=/gm) || []).length, result.columns.length);
      for (const c of result.columns) assert.ok(raw.includes(`pos=${c.pos} a=${c.a} b=${c.b} cin=${c.cin} -> `));
    }
  }
  const untrained = JSON.parse(call('web_infer', 'string', ['string', 'string', 'number'], ['9', '9', 1]));
  assert.equal(untrained.ok, true); assert.equal(untrained.valid, false); assert.ok(untrained.columns[0].ids.length);
  assert.equal(JSON.parse(call('web_infer', 'string', ['string', 'string', 'number'], ['10000000000', '1', 0])).answer, '10000000001');
  const long = JSON.parse(call('web_infer', 'string', ['string', 'string', 'number'], ['9'.repeat(1000), '1', 0]));
  assert.equal(long.answer, '1' + '0'.repeat(1000)); assert.equal(long.columns.length, 1000);
  assert.equal(call('web_predict','number',['string','string','number','number'],['9'.repeat(100),'1',0,0]),1);
  const direct = JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'})); assert.equal(direct.mode,'direct'); assert.ok(direct.rollovers > 0);
  const phases = [];
  for (let stage = 1; stage <= 6; stage++) {
    const budgets = full ? [200, 3000, 3000, 6000, 400, 2000] : [2, 2, 2, 2, 2, 2];
    assert.equal(call('web_begin', 'number', ['number', 'number', 'number', 'number'], [stage, budgets[stage - 1], stage === 6 ? 8 : 4, 0]), 1);
    let result = 0, oldMetric = -1;
    while (result === 0) {
      result = call('web_tick'); assert.ok(result >= 0);
      const s = status();
      if (full && s.metricStep !== oldMetric) {
        oldMetric = s.metricStep;
        console.log(`${variant} stage=${stage} step=${s.step} local=${s.correct}/200 margin=${s.margin.toFixed(3)}`);
      }
    }
    phases.push(status());
    if (full && stage >= 4) assert.equal(status().correct, 200, `stage ${stage} must certify`);
  }
  if (full) {
    assert.equal(call('web_certify', 'number', ['number'], [1]), 1);
    e.FS.writeFile('/export.cn', new Uint8Array());
    assert.equal(call('web_save', 'number', ['string', 'number'], ['/export.cn', 0]), 1);
    fs.writeFileSync(path.join(__dirname, 'build', `wasm-trained-${variant}-fp4.cn`), e.FS.readFile('/export.cn'));
  }
  const directPhases = [], chatPhases=[];
  if (!full) {
    assert.equal(call('web_init', 'number', ['number', 'number', 'number'], [24, 24, 20261018]), 1);
    assert.equal(call('web_protocol','number',['number','number'],[1,0]),1);
    assert.equal(call('web_mode', 'number', ['number'], [0]), 1);
    for (let stage = 1; stage <= 6; stage++) {
      assert.equal(call('web_begin', 'number', ['number', 'number', 'number', 'number'], [stage, 2, 1, 0]), 1);
      assert.equal(call('web_tick'), 0);
      assert.equal(call('web_tick'), 1);
      const s = status(); assert.equal(s.cot, false); assert.equal(s.step, 2);
      assert.ok(e.FS.readFile(`/experiment/stage-${stage}-last.cn`).length > 0);
      directPhases.push(s);
    }
    assert.equal(call('web_predict', 'number', ['string', 'string', 'number', 'number'], ['9', '9', 1, 0]), 1);
    const failed = JSON.parse(e.FS.readFile('/prediction.json', { encoding: 'utf8' }));
    assert.equal(failed.ok, true); assert.equal(failed.mode, 'direct');
    assert.ok(!failed.valid || failed.answer !== '18');
    for(const cot of [0,1]){
      assert.equal(call('web_init','number',['number','number','number'],[24,24,20261018]),1);
      assert.equal(call('web_protocol','number',['number','number'],[1,1]),1);
      assert.equal(call('web_mode','number',['number'],[cot]),1);
      const sample=JSON.parse(call('web_example','string',['string','string','number','number'],['99','1',cot,0]));
      assert.equal(sample.tokens.filter(t=>t===259).length,cot);
      assert.equal(sample.tokens.filter(t=>t===260).length,cot);
      assert.equal(sample.tokens.slice(0,sample.prompt).filter(t=>t===259||t===260).length,0);
      for(let stage=1;stage<=6;stage++){
        assert.equal(call('web_begin','number',['number','number','number','number'],[stage,1,1,0]),1);
        assert.equal(call('web_tick'),1);const s=status();assert.equal(s.protocol,'chat-v1');chatPhases.push(s);
      }
      assert.equal(call('web_predict','number',['string','string','number','number'],['99','1',1,cot]),1);
      const p=JSON.parse(e.FS.readFile('/prediction.json',{encoding:'utf8'}));
      assert.equal(p.protocol,'chat-v1');assert.ok(p.tokens.length>0);
      assert.ok(p.input_ids.includes(117));assert.equal(p.columns.length,0);
      fs.writeFileSync(path.join(__dirname,'build',`chat-smoke-${variant}-${cot?'cot':'direct'}.cn`),e.FS.readFile('/experiment/stage-6-last.cn'));
    }
  }
  console.log(`${variant}: loaded models, unrestricted/failed predictions and incremental CoT/direct stages passed`);
  return { variant, certs, phases, directPhases, chatPhases, unrestricted: { carryDigits: 1000, untrainedAccepted: true, directContextRollovers: direct.rollovers } };
}
(async () => {
  const reports = [];
  const scalarOnly = process.argv.includes('--scalar-pipeline');
  if (!scalarOnly) reports.push(await check('simd', process.argv.includes('--pipeline')));
  reports.push(await check('scalar', scalarOnly));
  const reportAt = process.argv.indexOf('--report');
  const report = reportAt < 0 ? (scalarOnly ? 'validation-scalar.json' : 'validation.json') : process.argv[reportAt + 1];
  fs.writeFileSync(path.join(__dirname, 'build', path.basename(report)), JSON.stringify({ passed: true, reports }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
