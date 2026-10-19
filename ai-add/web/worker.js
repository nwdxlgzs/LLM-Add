let engine, assets, active = false, paused = false, pipeline = [], runOptions, timer = 0;
const snapshots = new Map();
const protocolID = p => p === 'chat-v2' ? 3 : p === 'column-v2' ? 2 : p === 'chat-v1' ? 1 : 0;
let deployedCot = true, experiment = 0, runName = '', snapshotSerial = 0;
const invoke = (name, type, types = [], args = []) => engine.ccall(name, type, types, args);
const status = () => JSON.parse(invoke('web_status', 'string', ['number'], [1]));
function emit(event, data) { self.postMessage({ event, data }); }
function unpack(text) { return Uint8Array.from(atob(text), c => c.charCodeAt(0)); }
function putModel(name, path) {
  if (!assets[name]) throw new Error('找不到模型文件');
  engine.FS.writeFile(path, unpack(assets[name].base64));
}
function snapshot(label, path, metadata = {}) {
  if (!path) {
    path = '/snapshot.cn';
    if (!invoke('web_save', 'number', ['string', 'number'], [path, 1])) throw new Error('保存阶段模型失败');
  }
  const s = status(), item = { id: `snapshot:${Date.now()}-${++snapshotSerial}`, label: `${runName} · ${label}`,
    experiment, metric:s.stage===7?'greedy_final_answer':'teacher_forced_exact', cot: s.cot, protocol: s.protocol, stage: s.stage, step: s.step,
    correct: s.metricStep < 0 || (s.protocol === 'legacy-v1' && s.cot && s.stage >= 5 && label.includes('最后')) ? null : s.correct,
    bytes: engine.FS.readFile(path), ...metadata };
  snapshots.set(item.id, item); emit('snapshot', item); return item;
}
function predict(data) {
  const cot = data.cot ?? (data.student ? status().cot : deployedCot);
  if (!invoke('web_predict', 'number', ['string', 'string', 'number', 'number'], [data.a, data.b, data.student ? 1 : 0, cot ? 1 : 0]))
    throw new Error('请输入非空十进制自然数；若内存不足，可缩短输入重试。');
  const result = JSON.parse(engine.FS.readFile('/prediction.json', { encoding: 'utf8' }));
  const info = JSON.parse(invoke('web_status', 'string', ['number'], [data.student ? 1 : 0]));
  result.protocol = info.protocol; result.cache_bytes = info.cache_bytes;
  result.raw = engine.FS.readFile('/inference.txt', { encoding: 'utf8' }); return result;
}
function beginStage(stage) {
  const budget = runOptions.steps[stage - 1];
  if (!invoke('web_begin', 'number', ['number', 'number', 'number', 'number'],
    [stage, budget, stage === 6 ? runOptions.qatBatch : runOptions.batch, runOptions.lr || 0])) {
    active = false; pipeline = []; throw new Error('无法开始此阶段，请先结束当前实验或重置。');
  }
  emit('stageStart', status());
  const current = status();
  if (current.protocol !== 'legacy-v1') snapshot(`阶段 ${stage} · 训练前`, `/experiment/stage-${stage}-entry.cn`, { step: 0 });
  timer = setTimeout(pump, 0);
}
function pump() {
  if (!active || paused) return;
  try {
    const start = performance.now();
    let result = 0;
    do {
      result = invoke('web_tick', 'number');
      if (result < 0) throw new Error('训练更新失败，已停止；可以下载当前权重进行检查。');
    } while (!result && performance.now() - start < 35);
    const s = status();
    emit('progress', s);
    if (s.stage === 7 && invoke('web_rollouts','number')) emit('rollouts', JSON.parse(engine.FS.readFile('/rollouts.json',{encoding:'utf8'})));
    if (result) {
      snapshot(`阶段 ${s.stage} · 最后一次更新`, `/experiment/stage-${s.stage}-last.cn`, s.protocol !== 'legacy-v1' ? { correct: s.lastCorrect } : {});
      if (s.protocol !== 'legacy-v1' || (s.cot && s.stage >= 5)) snapshot(`阶段 ${s.stage} · 验证选择（第 ${s.bestStep || 0} 步）`, `/experiment/stage-${s.stage}.cn`, s.protocol !== 'legacy-v1' ? { step: s.bestStep } : {});
      emit('stageDone', s);
      if (pipeline.length) beginStage(pipeline.shift());
      else {
        active = false;
        emit('finished', status());
      }
    } else timer = setTimeout(pump, 0);
  } catch (error) {
    active = false; pipeline = []; emit('failure', error.message);
  }
}
self.onmessage = async ({ data: message }) => {
  const { id, command, data = {} } = message;
  try {
    let result;
    if (command === 'init') {
      assets = data.models;
      engine = await createEngine({ print: text => emit('log', text), printErr: text => emit('log', text) });
      if (!invoke('web_init_arch', 'number', ['number', 'number', 'number', 'number'], [data.config.dim, data.config.ff, data.config.layers || 6, data.config.seed]))
        throw new Error('引擎初始化失败');
      putModel(data.finalModel, '/builtin.cn');
      if (!invoke('web_load', 'number', ['string', 'number'], ['/builtin.cn', 0])) throw new Error('内置模型加载失败');
      invoke('web_protocol','number',['number','number'],[0,3]);
      result = { training: status(), deployed: JSON.parse(invoke('web_status', 'string', ['number'], [0])) };
    } else if (command === 'reset') {
      clearTimeout(timer); active = paused = false; pipeline = [];
      if (!invoke('web_init_arch', 'number', ['number', 'number', 'number', 'number'], [data.dim, data.ff, data.layers || 6, data.seed]))
        throw new Error('配置无效、超过 500,000 参数或内存不足；当前权重仍保留');
      if (!invoke('web_protocol','number',['number','number'],[1,protocolID(data.protocol || 'chat-v2')])) throw new Error('无法设置训练协议');
      result = status();
    } else if (command === 'shape') {
      result = invoke('web_shape','number',['number','number','number'],[data.dim,data.ff,data.layers||6]);
    } else if (command === 'load') {
      if (active && data.student) throw new Error('请先重置或结束训练，再导入模型');
      const saved = data.snapshot ? snapshots.get(data.snapshot) : null;
      if (data.snapshot && !saved) throw new Error('找不到阶段模型');
      const cot = saved ? saved.cot : data.cot !== false;
      const protocol = saved?.protocol || data.protocol || assets[data.name]?.protocol || (data.name?.startsWith('column-v2-') ? 'column-v2' : 'legacy-v1');
      if (saved) engine.FS.writeFile('/import.cn', saved.bytes);
      else if (data.bytes) engine.FS.writeFile('/import.cn', new Uint8Array(data.bytes));
      else putModel(data.name, '/import.cn');
      if (!invoke('web_load', 'number', ['string', 'number'], ['/import.cn', data.student ? 1 : 0]))
        throw new Error('文件格式或校验失败，请选择 Cairn .cn 检查点');
      if (!invoke('web_protocol', 'number', ['number','number'], [data.student ? 1 : 0, protocolID(protocol)]))
        throw new Error('新格式对话训练要求上下文至少 512 token（历史紧凑格式为 256）；请选择新建实验或相应模型包。');
      if (data.student) {
        if (!invoke('web_mode', 'number', ['number'], [cot ? 1 : 0])) throw new Error('无 CoT 训练需要上下文至少 40 的实验模型；请新建模型。');
      } else deployedCot = cot;
      result = JSON.parse(invoke('web_status', 'string', ['number'], [data.student ? 1 : 0]));
      result.cot = cot;
    } else if (command === 'run') {
      if (active) throw new Error('已有训练任务在运行');
      if (!invoke('web_mode', 'number', ['number'], [data.cot === false ? 0 : 1])) throw new Error('不能切换训练模式；无 CoT 旧模型请先新建实验。');
      if (status().parameters > 500000) throw new Error('训练上限为 500,000 参数；历史大模型仍可测试，请新建较小实验。');
      experiment++; runName = `实验 ${experiment} · ${status().protocol === 'column-v2' ? '结构化逐位' : ['chat-v1','chat-v2'].includes(status().protocol) ? '完整对话' : '旧协议'} · ${data.cot === false ? '无 CoT' : 'CoT'}`;
      snapshot('开始训练前');
      active = true; paused = false; runOptions = data;
      pipeline = data.all ? (status().protocol==='chat-v2'?[1,2,3,4,5,6,7]:[1,2,3,4,5,6]) : [data.stage];
      beginStage(pipeline.shift()); result = status();
    } else if (command === 'pause') {
      paused = !paused;
      if (!paused && active) timer = setTimeout(pump, 0);
      result = { paused, ...status() };
    } else if (command === 'infer') {
      if (data.cache !== undefined && !invoke('web_cache', 'number', ['number','number'], [data.student ? 1 : 0, data.cache ? 1 : 0])) throw new Error('无法准备推理权重缓存');
      result = predict(data);
    } else if (command === 'columnExample') {
      result=JSON.parse(invoke('web_column_example','string',['number','number','number','number'],[Number(data.a)%10,Number(data.b)%10,data.cin||0,data.copy?1:0]));
    } else if (command === 'columnExamples') {
      result=data.rows.map(row=>JSON.parse(invoke('web_column_example','string',['number','number','number','number'],[Number(BigInt(row.a)%10n),Number(BigInt(row.b)%10n),row.cin||0,data.copy?1:0])));
    } else if (command === 'example') {
      result = JSON.parse(invoke('web_example_format','string',['string','string','number','number','number'],[String(data.a),String(data.b),data.cot?1:0,data.copy?1:0,data.protocol==='chat-v1'?0:1]));
    } else if (command === 'examples') {
      result = data.rows.map(row => JSON.parse(invoke('web_example_format','string',['string','string','number','number','number'],[String(row.a),String(row.b),data.cot?1:0,data.copy?1:0,data.protocol==='chat-v1'?0:1])));
    } else if (command === 'restoreSnapshots') {
      for (const item of data.items) {
        snapshots.set(item.id, { ...item, bytes: new Uint8Array(item.bytes) });
        experiment = Math.max(experiment, item.experiment || Number(/^实验 (\d+)/.exec(item.label)?.[1]) || 0);
      }
      result = snapshots.size;
    } else if (command === 'exportSnapshot') {
      result = snapshots.get(data.id);
      if (!result) throw new Error('找不到阶段模型');
    } else if (command === 'deleteSnapshots') {
      if (active) throw new Error('请等待训练结束再删除检查点');
      for (const key of data.ids) snapshots.delete(key);
      result = snapshots.size;
    } else if (command === 'certify') {
      if (active) throw new Error('请先等待训练结束，再做文件认证');
      invoke('web_certify', 'number', ['number'], [data.student ? 1 : 0]);
      result = { report: JSON.parse(engine.FS.readFile('/certificate.json', { encoding: 'utf8' })),
        csv: engine.FS.readFile('/certificate.json.states.csv', { encoding: 'utf8' }) };
    } else if (command === 'export') {
      const path = data.stage ? `/experiment/stage-${data.stage}.cn` : '/download.cn';
      if (!data.stage && !invoke('web_save', 'number', ['string', 'number'], [path, data.optimizer ? 1 : 0]))
        throw new Error('保存失败');
      const bytes = engine.FS.readFile(path);
      self.postMessage({ id, ok: true, data: bytes }, [bytes.buffer]); return;
    } else if (command === 'evaluate') {
      if (active) throw new Error('请等待训练结束');
      const cot = data.cot ?? (data.student ? status().cot : deployedCot);
      const selectedInfo = JSON.parse(invoke('web_status','string',['number'],[data.student ? 1 : 0]));
      if (cot && selectedInfo.protocol === 'legacy-v1') {
        invoke('web_evaluate', 'number', ['number', 'number'], [data.student ? 1 : 0, 10000]);
        result = JSON.parse(engine.FS.readFile('/evaluation.json', { encoding: 'utf8' }));
      } else {
        const tests = {}; let state = 20261018;
        const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state; };
        for (const width of [1, 2, 4, 6, 10]) {
          let correct = 0;
          for (let i = 0; i < 20; i++) {
            const number = () => Array.from({ length: width }, () => random() % 10).join('');
            const a = number(), b = number(), prediction = predict({ a, b, student: data.student, cot });
            correct += prediction.valid && /^[0-9]+$/.test(prediction.answer) && BigInt(prediction.answer) === BigInt(a) + BigInt(b);
          }
          tests[`digits_${width}`] = { correct, total: 20 };
        }
        result = { passed: Object.values(tests).every(t => t.correct === t.total), tests, mode: cot ? 'cot' : 'direct', protocol: selectedInfo.protocol };
      }
    } else throw new Error('未知操作');
    self.postMessage({ id, ok: true, data: result });
  } catch (error) { self.postMessage({ id, ok: false, error: error.message }); }
};
