(() => {
  'use strict';
  // Capture the original DOM before inserting curves, states or stage cards.
  // Blob download works offline, including when opened from file://.
  const originalDocument = '<!doctype html>\n' + document.documentElement.outerHTML;
  const $ = id => document.getElementById(id);
  document.querySelectorAll('main pre,.reasoning-steps,.session-checkpoints').forEach(element => element.tabIndex = 0);
  const payload = JSON.parse($('payload').textContent);
  const engines = JSON.parse($('engines').textContent);
  const workerSource = JSON.parse($('worker-source').textContent);
  const names = ['数字 / 格式预训练', 'SFT · 一位无进位', 'SFT · 两位算式', 'SFT · 1/2/4/6/10 位课程', 'CoT RL · RLOO', 'FP4 · QAT', '自由草稿 RL · 仅最终答案'];
  const legacyNames = ['旧协议预训练', '旧协议 SFT 无进位', '旧协议 SFT 输出进位', '旧协议 SFT 输入进位', '旧协议 CoT RL', '旧协议 QAT'];
  const directNames = ['数字 / 格式预训练', 'SFT · 一位无进位', 'SFT · 两位算式', 'SFT · 1/2/4/6/10 位课程', '最终答案 RL · RLOO', 'FP4 · QAT', '自由草稿 RL · 仅最终答案'];
  const columnNames=['数字复制与角色格式','SFT · 无进位','SFT · 输出进位','SFT · 输入进位','逐位 CoT RL · RLOO','FP4 · QAT', '自由草稿 RL · 仅最终答案'];
  const dialogue = p => p === 'chat-v1' || p === 'chat-v2';
  const modern = p => dialogue(p) || p === 'column-v2';
  const structured = s => s.protocol === 'column-v2' && s.cot !== false;
  const stageName = s => (!modern(s.protocol) ? legacyNames : s.cot === false ? directNames : structured(s) ? columnNames : names)[s.stage-1];
  function syncTrainingMode(cot) {
    $('train-cot').checked = cot;
    const labels = modern(latest.protocol) ? (!cot ? directNames : latest.protocol==='column-v2' ? columnNames : names) : legacyNames;
    Array.from($('stage').options).forEach((option, i) => { option.textContent = `0${i + 1} · ${labels[i]||'仅新整式对话支持自由草稿 RL'}`;option.disabled=i===6&&latest.protocol!=='chat-v2'; });
    if (latest.protocol!=='chat-v2' && Number($('stage').value)===7) $('stage').value=6;
    $('metric-correct').previousElementSibling.textContent = Number($('stage').value)===7 ? '自由生成最终答案' : structured({...latest,cot}) ? '逐位对话完整匹配' : dialogue(latest.protocol) || modern(latest.protocol)&&!cot ? '固定对话完整匹配' : cot ? '旧局部预测正确数' : '固定整式正确数';
    window.dispatchEvent(new CustomEvent('aiadd:train-mode', { detail: cot }));
  }
  const stageFiles = Array.from({ length: payload.dialogue_stages.length }, (_, i) => `${payload.model_prefix}-stage-${i + 1}.cn`);
  const explanations = [
    '完整 user/assistant 消息，学习复制数字，尚未使用加法标签。', '一位无进位算式，assistant 输出一个思考块和最终答案。',
    '混合一位和两位算式，连续生成进位步骤与最终答案。', '逐步解锁长度并重放旧题；后半程混合 1/2/4/6/10 位。',
    '同题采样四个完整回答，以模型生成的步骤和答案奖励做 RLOO。', '执行真实 FP4 QAT，保留最后更新和验证选择；选择的版本可能为 FP32。', '不检查草稿的算法、字段或中间答案；仅对话控制符合法且最终答案正确奖励 1，真实自由生成选模。'];
  const recordedSteps = payload.dialogue_stages.map(s => s.steps);
  const recordedCorrect = payload.dialogue_stages.map(s => s.correct);
  let worker, ready = false, training = false, paused = false, busy = 0, serial = 0;
  let toastTimer, chartMode = 'structured', latest = {}, live = [], liveOffset = 0, previousStage = 0;
  let history = [], logs = [], activeName = stageFiles.at(-1);
  let lastMetric = '', latestFullTest = null;
  let lastAttempt = null, archiveDB = null;
  const archive = new Map();
  let deletedSnapshots = [];
  const pending = new Map();
  const number = (id, min, max) => {
    const n = Number($(id).value);
    if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${$(id).previousElementSibling?.textContent || id} 的范围应为 ${min}–${max}`);
    return n;
  };
  const sourceName = () => $('infer-model').value;
  const isStudent = () => sourceName() === 'student';
  const toast = text => {
    clearTimeout(toastTimer); $('toast').textContent = text; $('toast').classList.remove('hidden');
    toastTimer = setTimeout(() => $('toast').classList.add('hidden'), 5500);
  };
  function log(text) {
    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    logs.push(`${time}  ${text}`);
    if (logs.length > 1500) logs.shift();
    $('log').textContent = logs.join('\n'); $('log').scrollTop = $('log').scrollHeight;
  }
  function buttons() {
    const blocked = !ready || busy > 0;
    ['train-all', 'train-stage', 'create-model', 'import', 'stage', 'layers', 'dim', 'ff', 'seed'].forEach(id => $(id).disabled = blocked || training);
    ['pretrain-steps', 'sft-steps', 'sft2-steps', 'sft3-steps', 'rl-steps', 'qat-steps', 'outcome-steps', 'batch', 'lr', 'train-cot', 'train-protocol'].forEach(id => $(id).disabled = blocked || training);
    ['builtin-variant', 'infer-model', 'save-model', 'save-checkpoint'].forEach(id => $(id).disabled = blocked);
    $('infer').disabled = blocked || (training && isStudent());
    ['operand-a', 'operand-b', 'question-width', 'random-question', 'infer-style', 'infer-cache'].forEach(id => $(id).disabled = blocked);
    document.querySelectorAll('[data-a]').forEach(button => button.disabled = blocked);
    $('evaluate').disabled = blocked || training;
    $('pause').disabled = blocked || !training;
    $('reset').disabled = blocked;
    document.querySelectorAll('.load-stage,.resume-snapshot').forEach(b => b.disabled = blocked || training);
    document.querySelectorAll('.test-snapshot,.download-snapshot').forEach(b => b.disabled = blocked);
    document.querySelectorAll('.select-snapshot').forEach(b => b.disabled = blocked || training);
    $('archive-select-all').disabled = blocked || training || !archive.size;
    $('archive-delete').disabled = blocked || training || !selectedSnapshots().length;
    $('archive-undo').disabled = blocked || training || !deletedSnapshots.length;
  }
  async function task(fn) {
    busy++; buttons();
    try { return await fn(); }
    catch (error) { toast(error.message); log(`! ${error.message}`); throw error; }
    finally { busy--; buttons(); }
  }
  const action = (id, fn) => $(id).addEventListener('click', () => task(fn).catch(() => {}));
  function rpc(command, data = {}) {
    const id = ++serial;
    return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ id, command, data }); });
  }
  function download(name, content, type = 'application/octet-stream') {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement('a'); a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function unpack(text) { return Uint8Array.from(atob(text), c => c.charCodeAt(0)); }
  function config() { return { layers:number('layers',6,10), dim: number('dim', 16, 128), ff: number('ff', 16, 512), seed: number('seed', 1, 2147483647), protocol:'chat-v2' }; }
  function trainingOptions(all) {
    const sft = number('sft-steps', 1, 100000);
    return { all, stage: number('stage', 1, 7), batch: number('batch', 1, 32), qatBatch: number('batch', 1, 32),
      cot: $('train-cot').checked, lr: Number($('lr').value), steps: [number('pretrain-steps', 1, 100000), sft, number('sft2-steps',1,100000), number('sft3-steps',1,100000),
        number('rl-steps', 1, 100000), number('qat-steps', 1, 100000), number('outcome-steps', 1, 100000)] };
  }
  function modelInfo(info) {
    renderTopology(info);
    $('model-info').textContent = `${info.parameters.toLocaleString()} parameters · ${info.layers} layers · dim ${info.dim}`;
    $('config-code').textContent = info.parameters>500000 ? `// 历史大模型：允许测试，新训练限制为 500000\ncn_model *m = cn_load("checkpoint.cn", error, sizeof(error));\nconst cn_config *c = cn_model_config(m);` : `cn_config c;\nadd_chat_config(&c, ${$('seed').value});\nif (!add_chat_architecture(&c, ${info.layers}, ${info.dim}, ${info.ff})) return 0;\n// helper 设置完整的逐层 KV / 索引来源\n// CED ${info.encoder} + ${info.layers-info.encoder}; MoE 2+1; mHC 2 streams\n// heads ${info.heads} × ${info.headDim}, Q/O rank ${info.qRank}/${info.oRank}\n// Engram layer 1, 2/3/4-gram, head dim ${info.engramDim}\ncn_model *m = cn_model_create(&c, error, sizeof(error));`;

  }
  function renderTopology(info) {
    const groups=document.querySelectorAll('.model-group');
    for(let g=0;g<2;g++) {
      const container=groups[g].querySelector('.layers');container.replaceChildren();
      groups[g].querySelector('.group-label span:last-child').textContent=`× ${g?info.layers-info.encoder:info.encoder}`;
      for(let i=g?info.encoder:0;i<(g?info.layers:info.encoder);i++) {
        const owner=i<info.encoder?1:info.encoder;
        const type=i===0?'SWA':i===1?'FULL + ENGRAM':i===info.encoder?'FULL':i===info.layers-1?'REINDEX':'REUSE';
        const button=document.createElement('button');button.className='layer';
        const number=document.createElement('b');number.textContent=String(i).padStart(2,'0');
        const label=document.createElement('span');label.textContent=type;button.append(number,label);
        button.addEventListener('click',()=>{
          document.querySelectorAll('.layer').forEach(b=>b.classList.toggle('selected',b===button));
          $('layer-info').textContent=`Layer ${i} · ${type} · ${i===0?'512 token 因果滑窗':i===1?'压缩 KV 与稀疏索引，叠加 Engram 记忆':i===info.encoder?'KV 来自最后编码器状态':`复用第 ${owner} 层 KV${type==='REINDEX'?'，在本层重新建立索引':'及稀疏位置'}`}。每层包含 MoE 与双流 mHC。`;
        });container.append(button);
      }
    }
  }
  function renderRollouts(trace, label) {
    $('rollouts').replaceChildren();
    if(!trace?.samples) return;
    const specials={256:'<|im_start|>',257:'<|im_end|>',258:'<|endoftext|>',259:'<think>',260:'</think>'};
    $('rollout-note').textContent=`${label} · ${trace.a}+${trace.b} · 同题四次无约束采样，仅按最终答案与消息控制符奖励；草稿的内容不打分。`;
    trace.samples.forEach((sample,i)=>{
      const card=document.createElement('article');card.className='rollout';
      const h=document.createElement('h4');h.textContent=`采样 ${i+1} · reward ${sample.reward}`;
      const pre=document.createElement('pre');pre.tabIndex=0;
      pre.textContent=sample.tokens.map(t=>specials[t]??String.fromCharCode(t)).join('');
      card.append(h,pre);$('rollouts').append(card);
    });
  }
  $('rollouts-builtin').addEventListener('click',()=>renderRollouts(payload.outcome_rollouts,'预置阶段 7 · 最后一轮实际采样'));
  renderRollouts(payload.outcome_rollouts,'预置阶段 7 · 最后一轮实际采样');
  if(payload.outcome_comparison) {
    const table=document.createElement('table'),head=document.createElement('thead'),body=document.createElement('tbody');
    const tr=document.createElement('tr');
    for(const title of ['架构','参数','答案 / 200','1 / 2 / 4 / 6 / 10 位（各40题）','训练与评测耗时']){const th=document.createElement('th');th.textContent=title;tr.append(th);}head.append(tr);
    for(const item of payload.outcome_comparison.candidates){
      const row=document.createElement('tr');
      for(const value of [`${item.layers} 层 × ${item.dim} / FFN ${item.ff}${item.name===payload.outcome_comparison.winner?' · 采用':''}`,item.parameters.toLocaleString(),`${item.correct}/200`,item.width_correct.join(' / '),`${(item.seconds/60).toFixed(1)} 分钟`]){const td=document.createElement('td');td.textContent=value;row.append(td);}body.append(row);
    }
    table.append(head,body);$('depth-comparison').replaceChildren(table);
    const note=document.createElement('p');note.className='help';note.textContent='比较预算一致：300 / 1200 / 2000 / 4000 次更新，batch 4，相同种子和课程。按实际自由生成的最终答案选择；同分时选耗时较短者。这是本轮预算下的结果。赢家随后保留试训权重和优化器，继续混合长度训练至 8000 步，再完成阶段 5–7。';$('depth-comparison').append(note);
  }

  for(const id of ['layers','dim','ff'])$(id).addEventListener('change',async()=>{
    if(!ready)return;
    const count=await rpc('shape',config());
    $('shape-note').textContent=count?`待新建配置：${count.toLocaleString()} / 500,000 参数`:'此配置无效或超过 500,000 参数，请减小宽度或层数。';
  });
  function syncMyModels() {
    const empty = !$('my-models').children.length;
    $('my-models').hidden = empty; $('my-models').disabled = empty;
  }
  function ensureCurrentModel() {
    if (!$('current-model')) {
      const option = document.createElement('option'); option.id = 'current-model';
      option.value = 'student'; option.textContent = '当前训练模型（内存中）';
      $('my-models').prepend(option);
    }
    syncMyModels();
  }
  function clearExperiment(info, showCurrent = true) {
    training = paused = false; live = []; history = []; liveOffset = 0; previousStage = 0; lastMetric = '';
    latest = info;
    if (showCurrent) ensureCurrentModel();
    $('train-protocol').querySelector('[data-historical]')?.remove();
    if(info.protocol !== 'chat-v2') {
      const option=document.createElement('option');option.value=info.protocol;option.dataset.historical='1';
      option.textContent=`当前为旧检查点 ${info.protocol}（新建模型使用整式对话）`;
      $('train-protocol').append(option);
    }
    $('train-protocol').value=info.protocol;
    window.dispatchEvent(new CustomEvent('aiadd:protocol', { detail: info.protocol }));
    document.querySelectorAll('.stage').forEach(e => e.classList.remove('live', 'complete'));
    $('train-progress').style.width = '0%'; $('train-count').textContent = '0 updates';
    $('train-label').textContent = '实验模型已随机初始化，等待开始';
    $('metric-loss').textContent = '—'; $('metric-correct').textContent = '— / 200';
    $('metric-margin').textContent = '—'; $('metric-time').textContent = '0.0s';
    $('pause').textContent = '暂停'; modelInfo(info);
    if (isStudent()) clearAnswer();
    syncTrainingMode(info.cot !== false); drawChart(); buttons();
  }
  function progress(s) {
    latest = s;
    previousStage = s.stage;
    $('metric-loss').textContent = s.step ? s.loss.toFixed(5) : '—';
    $('metric-correct').textContent = `${s.correct} / 200`;
    $('metric-correct').previousElementSibling.textContent = structured(s) ? '逐位对话完整匹配' : s.stage===7 ? '自由生成最终答案' : modern(s.protocol) ? '固定对话完整匹配' : s.cot === false ? '固定整式正确数' : '旧局部预测正确数';
    $('metric-margin').textContent = modern(s.protocol) && !structured(s) || s.cot === false ? '—' : s.margin.toFixed(3);
    if (s.stage===7) {
      $('metric-margin').previousElementSibling.textContent = '本轮平均奖励';
      $('metric-margin').textContent = s.reward.toFixed(3);
    } else if (modern(s.protocol) && !structured(s)) {
      $('metric-margin').previousElementSibling.textContent = '验证 Token 正确率';
      $('metric-margin').textContent = s.tokenTotal ? `${(100 * s.tokenCorrect / s.tokenTotal).toFixed(1)}%` : '—';
    } else $('metric-margin').previousElementSibling.textContent = '最小间隔';
    $('metric-time').textContent = `${s.seconds.toFixed(1)}s`;
    $('train-count').textContent = `${s.step} / ${s.total}`;
    $('train-progress').style.width = `${Math.min(100, s.step / s.total * 100)}%`;
    $('train-label').textContent = `${stageName(s)}${paused ? ' · 已暂停' : ' · 训练中'}`;
    $('metric-note').textContent = s.stage===7 ? `只看最终答案，草稿不评分：1/2/4/6/10 位各 40 道真实贪心生成 [${(s.widthCorrect||[]).join(', ')}]；选择第 ${s.bestStep} 步，末次 ${s.lastCorrect}/200；采样均奖不更新，有奖励差异 ${s.informative} 次。` : structured(s) ? `200 种 a/b/cin 的真实逐位对话，含完整结束标记；最小间隔 ${s.margin.toFixed(3)}，CE ${s.validationLoss.toFixed(5)}。位置由调度器处理，数字与进位均由网络预测。验证选择第 ${s.bestStep} 步；RL 有奖励差异 ${s.informative} 次。` : modern(s.protocol) ? `固定验证：1/2/4/6/10 位各 40 题，整段匹配 [${(s.widthCorrect || []).join(', ')}]；验证 CE ${s.validationLoss.toFixed(4)}。当前第 ${s.step} 步，验证选择第 ${s.bestStep} 步；RLOO 有奖励差异 ${s.informative} 次。同奖励组不更新权重；预训练按复制目标训练并保留最后更新。` : `${s.cot === false ? '旧协议 200 道整式' : '旧协议 200 种逐位状态'}最近检查于本阶段第 ${s.metricStep} 步；当前第 ${s.step} 步。RLOO 有奖励差异的更新：${s.informative}。不同协议的指标分别比较。`;
    const key = `${s.stage}:${s.metricStep}`;
    if (s.step > 0 && s.step === s.metricStep && key !== lastMetric) {
      lastMetric = key;
      const point = { x: liveOffset + s.step, step: s.step, stage: s.stage, correct: s.correct, loss: s.loss, margin: s.margin, seconds: s.seconds };
      live.push(point); history.push({ ...s });
      const metric = s.stage===7 ? '最终答案' : structured(s) ? '逐位对话' : modern(s.protocol) ? '完整对话' : s.cot === false ? '整式' : '旧局部状态';
      log(`${stageName(s)}  step=${s.step}  loss=${s.loss.toFixed(5)}  ${metric}=${s.correct}/200${(!modern(s.protocol)||structured(s)) && s.cot !== false ? `  margin=${s.margin.toFixed(3)}` : ''}`);
      if (chartMode === 'live') drawChart();
    }
  }
  function clearAnswer() {
    lastAttempt = null; $('download-attempt').disabled = true;
    $('answer-check').textContent = ''; $('expected-answer').textContent = '';
    $('answer').textContent = '等待模型作答'; $('infer-timing').textContent = '待计算';
    $('answer-note').textContent = '点击“让模型作答”，查看本次实际生成的回答。';
    $('raw-transcript').textContent = `<|im_start|>user\n${$('operand-a').value}+${$('operand-b').value}\n<|im_end|>\n\n等待模型输出。`;
    $('trace').textContent = '等待这道题的模型预测。'; $('raw-calls').textContent = '等待这道题的模型预测。';
  }
  async function infer() {
    const a = $('operand-a').value.trim(), b = $('operand-b').value.trim();
    if (!/^[0-9]+$/.test(a) || !/^[0-9]+$/.test(b)) throw new Error('请输入非空的十进制自然数，不限制位数。');
    clearAnswer();
    $('answer').textContent = '模型计算中…';
    const start = performance.now();
    const style = $('infer-style').value;
    const result = await rpc('infer', { a, b, student: isStudent(), cache: $('infer-cache').value === 'decoded', ...(style === 'auto' ? {} : { cot: style === 'cot' }) });
    const expected = decimalSum(a, b), correct = result.valid && /^[0-9]+$/.test(result.answer) && result.answer.replace(/^0+(?=\d)/, '') === expected;
    lastAttempt = { a, b, source: sourceName(), variant:sourceName()===stageFiles.at(-1)?$('builtin-variant').value:null, expected, correct, ...result };
    $('download-attempt').disabled = false;
    $('answer-check').className = `answer-check ${correct ? 'pass' : 'fail'}`;
    const errors = { digit_token: '结果位不是数字', carry_token: '进位不是 0 或 1', end_token: '缺少 </think>', answer_token: '答案中出现非数字 token', generation_limit: '达到本题生成预算仍未结束', column_format: '逐位回答未符合 digit 空格 carry 与消息结束格式', empty_answer: '空答案', forward: '前向计算失败', chat_end: '缺少 assistant 消息结束标记', think_start: '缺少 assistant 思考开始标记', think_end: '缺少思考结束标记', think_format: '思考内容中出现非法标记', scratchpad_control:'草稿中出现错误的控制符' };
    $('answer-check').textContent = correct ? '✓ 计算正确' : result.valid ? '✗ 计算错误，保留模型原答案' : `✗ ${result.message_complete ? '输出格式错误' : '输出未完成'}：${errors[result.error] || result.error}`;
    $('expected-answer').textContent = `正确答案（独立校验）：${expected}`;
    $('answer').textContent = result.answer || '未生成完整数字答案';
    $('infer-timing').textContent = `${(performance.now() - start).toFixed(0)} ms`;
    $('answer-note').textContent = `${result.protocol === 'column-v2' ? result.mode==='cot' ? '结构化逐位对话（调度器取位与拼接，网络预测数字与进位）' : '完整对话直接回答' : dialogue(result.protocol) ? '完整对话自回归' : '旧协议'} · ${result.mode === 'cot' ? 'CoT' : '直接回答'} · 输入 ${a.length} / ${b.length} 位 · 生成 ${result.generated} token${result.rollovers ? ` · 上下文滚动 ${result.rollovers} 次` : ''}${result.cache_bytes ? ` · 解码权重缓存 ${(result.cache_bytes / 1024).toFixed(0)} KiB` : ''}。正确答案仅作校验。`;
    $('raw-transcript').textContent = result.raw;
    $('trace').replaceChildren();
    $('trace').classList.toggle('chat-reasoning', dialogue(result.protocol) || result.mode === 'direct');
    const calls = [], positions = ['个位', '十位', '百位', '千位', '万位', '十万位', '百万位', '千万位', '亿位', '十亿位'];
    for (const col of result.columns.slice(0, 200)) {
      const ids = col.ids, malformed = result.protocol==='column-v2' ? !col.valid : col.digit < 0 || col.cout < 0 || ids[2] !== 260;
      const item = document.createElement('div'); item.className = `reasoning-step${malformed ? ' invalid' : ''}`;
      const position = document.createElement('span'); position.className = 'position'; position.textContent = positions[col.pos] || `第 ${col.pos + 1} 位`;
      const sentence = document.createElement('p');
      sentence.textContent = malformed ? `输入 ${col.a}、${col.b}、进位 ${col.cin}；模型生成 token ID [${ids.join(', ')}]，不符合逐位格式，停止在这里。` : `读入 ${col.a} 和 ${col.b}，接收进位 ${col.cin}。模型预测这一位写 ${col.digit}，${col.cout ? '向下一位进 1' : '不向下一位进位'}。`;
      item.append(position, sentence); $('trace').append(item);
      if(result.protocol==='column-v2')calls.push(`第 ${col.pos} 列（pos 是调度元数据）\n实际输入 ID: [${col.input_ids.join(', ')}]\n实际生成 ID: [${ids.join(', ')}]`);
      else calls.push(`第 ${col.pos} 列\n输入: <|im_start|>${col.a}${col.b}${col.cin}<think>\n输入 ID: [256, ${48 + col.a}, ${48 + col.b}, ${48 + col.cin}, 259]\n实际生成 ID: [${ids.join(', ')}]`);
    }
    if (dialogue(result.protocol) || result.protocol === 'column-v2' && result.mode === 'direct') {
      const start = result.tokens.indexOf(259), end = result.tokens.indexOf(260, start + 1);
      const special = {256:'<|im_start|>',257:'<|im_end|>',258:'<|endoftext|>',259:'<think>',260:'</think>'};
      if (start >= 0) result.reasoning = result.tokens.slice(start+1,end<0?undefined:end).map(t=>special[t]??String.fromCharCode(t)).join('');
      $('trace').textContent = result.mode === 'cot' ? result.reasoning || (start>=0&&end>start?'模型生成了空草稿，直接给出最终答案。':'模型未生成完整的思考块；所有实际 token 保留在原始输出中。') : '整段 assistant 直接回答，不要求生成思考块。';
      calls.push(`实际完整对话输入 ID:\n[${result.input_ids.join(', ')}]\n\n模型连续生成 ID:\n[${result.tokens.join(', ')}]\n上下文滚动次数：${result.rollovers}`);
      calls.push(`stop=${result.stop}; message_complete=${result.message_complete}; format=${result.error || 'valid'}`);
    } else if (result.mode === 'direct') {
      $('trace').textContent = '这个模型直接生成答案，没有训练或输出逐位思维链。实际生成（包括非法 token）见原始文本。';
      calls.push(`整式输入: <|im_start|>${a}+${b}=\n上下文滚动次数: ${result.rollovers}\n没有进位标签，也没有逐位加法器兜底。`);
    } else if (result.columns.length > 200) {
      const note = document.createElement('p'); note.textContent = `页面展开前 200 列，共 ${result.columns.length} 列；原始文本和下载包含全部预测。`; $('trace').append(note);
    }
    const final = document.createElement('div'); final.className = 'reasoning-final';
    final.textContent = result.valid ? `模型输出已结束。${correct ? '与正确答案一致。' : '与正确答案不一致，保留本次错误结果。'}` : '格式错误也属于实验结果，本次所有实际预测已保留。';
    $('trace').append(final); $('raw-calls').textContent = calls.join('\n\n');
  }
  function decimalSum(a, b) {
    const result = []; let i = a.length - 1, j = b.length - 1, carry = 0;
    while (i >= 0 || j >= 0 || carry) { const n = (i >= 0 ? Number(a[i--]) : 0) + (j >= 0 ? Number(b[j--]) : 0) + carry; result.push(String(n % 10)); carry = Math.floor(n / 10); }
    return result.reverse().join('').replace(/^0+(?=\d)/, '') || '0';
  }
  const columnRecorded=[]; {let offset=0,last=0,stage=0; for(const line of payload.dialogue_training.trim().split(/\r?\n/).slice(1)){const fields=line.split(','),next=Number(fields[0]);if(stage&&next!==stage)offset+=last;stage=next;last=Number(fields[1]);columnRecorded.push({x:offset+last,stage,correct:Number(fields[3]),loss:Number(fields[2])});}}
  const recorded = [];
  {
    let offset = 0, oldStage = '', oldStep = 0;
    for (const file of ['metrics.csv', 'qat-metrics.csv']) {
      for (const line of payload.evidence[file].trim().split(/\r?\n/).slice(1)) {
        const [stage, step, loss, correct] = line.split(',');
        if (oldStage && oldStage !== stage) offset += oldStep;
        oldStage = stage; oldStep = Number(step);
        recorded.push({ x: offset + Number(step), loss: Number(loss), correct: Number(correct) });
      }
    }
  }
  function drawChart() {
    const canvas = $('chart'), width = canvas.clientWidth, height = 190, ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = height * ratio;
    const c = canvas.getContext('2d'); c.scale(ratio, ratio);
    const points = chartMode === 'recorded' ? recorded : chartMode==='structured' ? columnRecorded : live;
    const l = 31, r = width - 23, t = 13, b = height - 27;
    c.font = '9px Consolas, monospace'; c.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = b - (b - t) * i / 4;
      c.strokeStyle = '#e8ece3'; c.lineWidth = 1; c.beginPath(); c.moveTo(l, y); c.lineTo(r, y); c.stroke();
      c.fillStyle = '#9aa58f'; c.fillText(`${i * 25}%`, l - 6, y + 3);
    }
    if (!points.length) { c.fillStyle = '#a0ad94'; c.textAlign = 'center'; c.fillText('开始一次训练，在这里看到真实学习曲线', width / 2, 95); return; }
    const maxX = Math.max(points[points.length - 1].x, 1), maxLoss = Math.max(...points.map(p => p.loss), .1);
    const x = p => l + p.x / maxX * (r - l), y = p => b - p.correct / 200 * (b - t);
    // The new stage changes both the reward and the validation metric. Avoid
    // drawing a connecting slope that looks like a gain or a regression.
    const groups=[];
    for(const p of points) {
      if(!groups.length || (p.stage===7)!==(groups.at(-1)[0].stage===7)) groups.push([]);
      groups.at(-1).push(p);
    }
    const gradient = c.createLinearGradient(0, t, 0, b); gradient.addColorStop(0, '#dcebd473'); gradient.addColorStop(1, '#edf4e91a');
    for(const group of groups) {
      c.beginPath();c.moveTo(x(group[0]),b);for(const p of group)c.lineTo(x(p),y(p));c.lineTo(x(group.at(-1)),b);c.closePath();c.fillStyle=gradient;c.fill();
      c.beginPath();group.forEach((p,i)=>i?c.lineTo(x(p),y(p)):c.moveTo(x(p),y(p)));c.strokeStyle='#3e8052';c.lineWidth=2;c.stroke();
      c.beginPath();group.forEach((p,i)=>{const v=b-p.loss/maxLoss*(b-t);i?c.lineTo(x(p),v):c.moveTo(x(p),v);});c.strokeStyle='#b7a06c';c.lineWidth=1.3;c.stroke();
    }
    if(groups.length>1) {
      const boundary=x(groups.at(-1)[0]);c.setLineDash([3,3]);c.strokeStyle='#9aa58f';c.beginPath();c.moveTo(boundary,t);c.lineTo(boundary,b);c.stroke();c.setLineDash([]);
      c.fillStyle='#62705d';c.textAlign='right';c.fillText('阶段 7 · 最终答案',Math.min(r,boundary+45),t+9);
    }
    c.fillStyle = '#9aa58f'; c.textAlign = 'center';
    for (let i = 0; i <= 4; i++) c.fillText(Math.round(maxX * i / 4).toLocaleString(), l + (r - l) * i / 4, height - 10);
  }
  function setChart(mode) {
    chartMode = mode; $('chart-recorded').classList.toggle('active', mode === 'recorded');
    $('chart-live').classList.toggle('active', mode === 'live'); $('chart-structured').classList.toggle('active',mode==='structured');
    $('chart-subtitle').textContent = mode === 'recorded' ? '/ 旧局部模型历史（指标不同）' : mode==='structured' ? '/ 本轮整式对话 · 原生 C 实训' : '/ 当前浏览器实际训练'; drawChart();
  }
  function rememberSnapshot(item, persist = true) {
    if (archive.has(item.id)) return;
    archive.set(item.id, item);
    const displayLabel = `${item.protocol === 'chat-v2' ? '' : item.protocol === 'chat-v1' ? '历史紧凑格式 · ' : item.protocol === 'column-v2' ? '旧逐列实验 · ' : '旧协议 · '}${item.label}`;
    const option = document.createElement('option'); option.value = item.id; option.textContent = displayLabel;
    $('my-models').append(option); syncMyModels();
    const row = document.createElement('div'); row.className = 'archive-row'; row.dataset.snapshot = item.id;
    const choice = document.createElement('input'); choice.type = 'checkbox'; choice.className = 'select-snapshot';
    choice.setAttribute('aria-label', `选择 ${displayLabel}`); choice.addEventListener('change', archiveSelection);
    const header = document.createElement('label'); header.className = 'archive-label';
    const label = document.createElement('span'); label.textContent = `${displayLabel}${item.correct == null ? '' : ` · ${item.correct}/200${item.metric==='greedy_final_answer'?' 最终答案':''}`}`;
    header.append(choice, label); row.append(header);
    for (const [text, cls, fn] of [
      ['测试', 'test-snapshot', async () => { $('infer-model').value = item.id; await selectModel(); location.hash = 'play'; await infer(); }],
      ['继续训练', 'resume-snapshot', async () => {
        const info = await rpc('load', { snapshot: item.id, student: true }); clearExperiment(info);
        syncTrainingMode(item.cot); $('dim').value = info.dim; $('ff').value = info.ff; $('layers').value = info.layers;
        $('stage').value = Math.min(latest.protocol==='chat-v2'?7:6, item.stage + 1);
        $('train-label').textContent = `已载入：${item.label}`; location.hash = 'lab';
      }],
      ['下载模型包', 'download-snapshot', async () => download(`${item.id.replace(':', '-')}.aiadd`, JSON.stringify({ format: 'aiadd-experiment-v1', ...item, bytes: undefined, model_base64: pack(item.bytes) }), 'application/json')]
    ]) { const b = document.createElement('button'); b.className = cls; b.textContent = text; b.addEventListener('click', () => task(fn).catch(() => {})); row.append(b); }
    $('session-checkpoints').append(row);
    if (persist && archiveDB) {
      try { const tx = archiveDB.transaction('models', 'readwrite'); tx.objectStore('models').put(item); tx.onabort = () => $('archive-status').textContent = '本机存储失败；当前页面仍保留模型，请下载模型包。'; }
      catch { $('archive-status').textContent = '本机存储不可用；当前页面保留全部模型，请下载模型包。'; }
    }
    archiveSelection();
  }
  function selectedSnapshots() {
    return Array.from(document.querySelectorAll('.select-snapshot:checked'), input => input.closest('.archive-row').dataset.snapshot);
  }
  function archiveSelection() {
    const count = selectedSnapshots().length;
    $('archive-selected').textContent = `已选 ${count} / ${archive.size}`;
    $('archive-select-all').checked = !!archive.size && count === archive.size;
    $('archive-select-all').indeterminate = count > 0 && count < archive.size;
    buttons();
  }
  async function storeArchive(items, remove) {
    if (!archiveDB) return;
    await new Promise((resolve, reject) => {
      const tx = archiveDB.transaction('models', 'readwrite'), store = tx.objectStore('models');
      for (const item of items) remove ? store.delete(item.id) : store.put(item);
      tx.oncomplete = resolve; tx.onabort = tx.onerror = () => reject(tx.error || new Error('本机存储操作失败，模型未从列表移除。'));
    });
  }
  action('archive-delete', async () => {
    const items = selectedSnapshots().map(id => archive.get(id));
    if (!items.length) return;
    const selected = sourceName();
    await storeArchive(items, true);
    await rpc('deleteSnapshots', { ids: items.map(item => item.id) });
    const removed = new Set(items.map(item => item.id));
    for (const item of items) archive.delete(item.id);
    for (const row of document.querySelectorAll('.archive-row')) if (removed.has(row.dataset.snapshot)) row.remove();
    for (const option of Array.from($('infer-model').options)) if (removed.has(option.value)) option.remove();
    if (!archive.size) $('current-model')?.remove();
    syncMyModels();
    if (!archive.size || removed.has(selected)) {
      $('infer-model').value = stageFiles.at(-1); await selectModel();
    }
    deletedSnapshots = items; archiveSelection();
    $('archive-status').textContent = `已删除所选 ${items.length} 个本地检查点；关闭页面前可撤销本次删除。当前实验权重仍可继续训练。`;
    log(`手动删除 ${items.length} 个本地检查点。`);
  });
  action('archive-undo', async () => {
    const items = deletedSnapshots;
    await storeArchive(items, false);
    await rpc('restoreSnapshots', { items });
    items.forEach(item => rememberSnapshot(item, false)); deletedSnapshots = []; archiveSelection();
    $('archive-status').textContent = `已恢复 ${items.length} 个检查点。`;
  });
  $('archive-select-all').addEventListener('change', () => {
    document.querySelectorAll('.select-snapshot').forEach(input => input.checked = $('archive-select-all').checked);
    archiveSelection();
  });
  function pack(bytes) { let s = ''; for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode(...bytes.subarray(i, i + 8192)); return btoa(s); }
  async function restoreArchive() {
    try {
      archiveDB = await new Promise((resolve, reject) => { const req = indexedDB.open('aiadd-experiments-v1', 1); req.onupgradeneeded = () => req.result.createObjectStore('models', { keyPath: 'id' }); req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); });
      const items = await new Promise((resolve, reject) => { const req = archiveDB.transaction('models').objectStore('models').getAll(); req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); });
      await rpc('restoreSnapshots', { items }); items.forEach(item => rememberSnapshot(item, false));
      archiveSelection();
      $('archive-status').textContent = `已从本机恢复 ${items.length} 个阶段模型；新实验阶段会自动保存。也可下载模型包跨设备导入。`;
    } catch { $('archive-status').textContent = '浏览器未提供持久存储；阶段模型仍保留在当前页面，重置不会删除，请在关闭前下载模型包。'; }
  }
  async function selectModel() {
    const base=sourceName(),hasVariants=!!Object.keys(payload.variants||{}).length,variant=hasVariants?$('builtin-variant').value:'selected';
    $('builtin-variant-field').hidden=!hasVariants||base!==stageFiles.at(-1);
    const name=base===stageFiles.at(-1)&&variant!=='selected'?base.replace('.cn',`-${variant}.cn`):base;
    if (name !== 'student' && name !== activeName) {
      await rpc('load', { ...(archive.has(name) ? { snapshot: name } : { name }), student: false }); activeName = name;
    }
    clearAnswer(); buttons();
  }
  function onEvent(event, data) {
    if (event === 'rollouts') { renderRollouts(data, '当前实验 · 阶段 7 的四份实际采样'); return; }
    if (event === 'log') { if (data.trim()) log(data); return; }
    if (event === 'snapshot') { rememberSnapshot(data); return; }
    if (event === 'stageStart') {
      ensureCurrentModel();
      liveOffset = live.length ? live[live.length - 1].x : 0;
      lastMetric = '';
      training = true; paused = false; progress(data); buttons();
      if (isStudent()) {
        clearAnswer(); $('answer').textContent = '模型正在训练';
        $('answer-note').textContent = '训练结束后，用更新的权重生成本题答案。';
        $('trace').textContent = '训练结束后再生成当前模型的轨迹。';
      }
      document.querySelectorAll('.stage').forEach((e, i) => e.classList.toggle('live', i + 1 === data.stage));
      log(`→ 开始 ${stageName(data)}，最多 ${data.total} 步`);
    } else if (event === 'progress') progress(data);
    else if (event === 'stageDone') {
      const card = document.querySelectorAll('.stage')[data.stage - 1]; card.classList.remove('live'); card.classList.add('complete');
      log(`✓ 阶段完成：${stageName(data)}，${data.step} 步，${data.correct}/200${modern(data.protocol) ? `；最后更新 ${data.lastCorrect}/200，采用验证选择第 ${data.bestStep} 步` : ''}`);
    } else if (event === 'finished' || event === 'halted' || event === 'failure') {
      training = false; paused = false; buttons(); $('pause').textContent = '暂停';
      if (event === 'finished') {
        $('train-label').textContent = modern(data.protocol) ? `${stageName(data)} · 已结束 · 验证 ${data.correct}/200` : `${stageName(data)} · 已完成`;
        $('train-progress').style.width = '100%';
        $('train-count').textContent = `${data.step} updates${data.step < data.total ? ' · 提前达标' : ''}`;
        toast(modern(data.protocol) ? `阶段已结束，采用验证选择的权重（${data.correct}/200）；训练前、最后更新和验证选择均已保存。` : data.stage === 6 ? 'FP4 训练完成，可以选择“我的实验模型”试题。' : '本阶段训练完成。');
      } else { $('train-label').textContent = '已停止 · 可调整参数继续'; toast(data); log(`! ${data}`); }
    }
  }
  async function startWorker(variant) {
    const blob = new Blob([engines[variant], '\n', workerSource], { type: 'text/javascript' });
    const url = URL.createObjectURL(blob);
    worker = new Worker(url);
    worker.onmessage = ({ data }) => {
      if (data.event) return onEvent(data.event, data.data);
      const waiter = pending.get(data.id); if (!waiter) return;
      pending.delete(data.id); data.ok ? waiter.resolve(data.data) : waiter.reject(new Error(data.error));
    };
    worker.onerror = event => {
      const error = new Error(`浏览器引擎错误：${event.message || '请使用新版 Chrome、Edge、Firefox 或 Safari'}`);
      for (const waiter of pending.values()) waiter.reject(error); pending.clear();
      if (ready) { training = false; ready = false; buttons(); $('runtime-error').textContent = error.message; $('runtime-error').classList.remove('hidden'); }
    };
    try { return await rpc('init', { models: {...payload.models,...payload.variants}, config: payload.default_config, finalModel: stageFiles.at(-1) }); }
    finally { URL.revokeObjectURL(url); }
  }
  async function init() {
    try {
      let info;
      try { info = await startWorker('simd'); }
      catch { worker?.terminate(); log('SIMD 不可用，自动使用兼容标量引擎。'); info = await startWorker('scalar'); }
      ready = true; clearExperiment(info.training, false);
      $('dim').value=info.training.dim; $('ff').value=info.training.ff; $('layers').value=info.training.layers;
      window.dispatchEvent(new Event('aiadd:ready'));
      $('runtime').replaceChildren();
      const dot = document.createElement('i'); dot.className = 'dot'; const label = document.createElement('span');
      label.textContent = `${info.deployed.backend === 'wasm-simd128' ? 'SIMD128' : 'Scalar'} · 离线就绪`;
      $('runtime').append(dot, label);
      log(`引擎就绪：${info.deployed.backend} · ${info.training.parameters.toLocaleString()} 参数`);
      await restoreArchive();
      await task(infer);
    } catch (error) {
      $('runtime-error').textContent = `无法启动：${error.message}。请用新版桌面浏览器打开完整 ai-add.html 文件。`;
      $('runtime-error').classList.remove('hidden'); $('runtime').textContent = '引擎加载失败'; ready = false; buttons();
    }
  }
  // Static model journey, deliberately separate from this browser's training.
  $('builtin-size').textContent = (payload.models[stageFiles.at(-1)].bytes/1024).toFixed(1);
  const pilot=payload.capacity_pilots.find(p=>p.dim===payload.default_config.dim && p.ff===payload.default_config.ff);
  $('capacity-note').textContent=`当前预置模型 ${payload.default_config.layers} 层 / dim ${payload.default_config.dim} / FFN ${payload.default_config.ff}，${payload.dialogue_stages.at(-1).parameters.toLocaleString()} 参数。深宽对照详见下方记录。`;
  if(pilot) $('capacity-note').textContent=`本轮内置：dim ${pilot.dim} / FFN ${pilot.ff} / ${pilot.parameters.toLocaleString()} 参数；注意力每头 ${pilot.head_dim}，Q/O rank ${pilot.q_rank}。本机 C 十位样本反向 + 更新约 ${(pilot.measurements[2].seconds_per_sequence*1000).toFixed(0)} ms/序列；浏览器和不同 CPU 耗时会变化。`;
  $('quality-note').textContent = `预置最终选择：${payload.dialogue_stages.at(-1).parameters.toLocaleString()} 参数，独立十位随机测试：最终答案 ${payload.dialogue_test['10'].correct}/${payload.dialogue_test['10'].total}。下方可展开自由草稿 RL 前后和前轮模型的同题对照。结果来自实际连续生成，失败回答原样保留。`;
  const wasmScores=payload.wasm_validation?.reports.find(r=>r.variant==='simd')?.scores;
  $('engine-agreement').textContent=payload.outcome_parity ? `本轮 C / WASM：完成 ${payload.outcome_parity.results.length} 组数值对照；全部容差检查${payload.outcome_parity.passed?'通过':'存在差异，原始反例已保留'}。自由生成成绩单独统计，数值容差通过不保证长程训练轨迹相同。` : '历史量化后端存在 logits 与长序列 token 分歧，分后端报告真实成绩。';
  for(const [width,score] of Object.entries(payload.dialogue_test)){
    const old=(payload.outcome_before?payload.previous_scores:payload.capacity_baseline)[width],audit=payload.capacity_comparison?.groups[width]?.new, before=payload.outcome_before?.[width];
    const row=document.createElement('tr');
    const ws=wasmScores?.[width];
    const last=payload.outcome_last?.[width];
    for(const value of [width,`${old.correct}/${old.total}`,`${score.correct}/${score.total}`,ws?`${ws.correct}/${ws.total}`:'—',before?`${before.correct}/${before.total}`:'—',last?`${last.correct}/${last.total}`:'—',before?`${score.correct-before.correct>=0?'+':''}${score.correct-before.correct}`:'—']){
      const cell=document.createElement('td');cell.textContent=value;row.append(cell);
    }
    $('capacity-scores').append(row);
  }
  names.slice(0,payload.dialogue_stages.length).forEach((name, i) => {
    const card = document.createElement('article'); card.className = 'stage';
    card.innerHTML = `<span class="num">0${i + 1} / ${i === 5 ? 'QUANTIZE' : 'LEARN'}</span><h3>${name}</h3><p>${explanations[i]}</p><strong>${recordedCorrect[i]}<span class="sub"> / 200</span></strong><span class="smallprint">${i===6?'自由生成答案':'固定整段匹配'} · ${['FP32','FP8','FP4'][payload.dialogue_stages[i].precision]}<br>${recordedSteps[i].toLocaleString()} updates</span><br><button class="load-stage" disabled>载入此阶段继续 ↗</button>`;
    const view = document.createElement('button'); view.className = 'view-data'; view.textContent = '看本阶段数据';
    view.addEventListener('click', () => { window.dispatchEvent(new CustomEvent('aiadd:show-data', { detail: String(i + 1) })); location.hash = 'data'; });
    card.append(document.createElement('br'), view);
    card.querySelector('.load-stage').addEventListener('click', () => task(async () => {
      const info = await rpc('load', { name: stageFiles[i], student: true });
      clearExperiment(info); syncTrainingMode(true); $('dim').value = info.dim; $('ff').value = info.ff; $('layers').value = info.layers;
      $('stage').value = Math.min(7, i + 2); $('train-label').textContent = `已载入预训练阶段：${name}`;
      log(`载入 ${stageFiles[i]}，可从所选阶段继续训练。`); location.hash = 'lab';
    }).catch(() => {})); $('stages').append(card);
  });
  const layerDescriptions = [
    ['滑动窗口注意力', '本轮保持 512 token 窗口，容纳十位题的完整训练对话；扩容增加模块宽度。旧检查点保留自己的窗口。每层还包含 MoE 与双流 mHC。'],
    ['Full + Engram', '自建全局 KV / 索引，按 2 个 token 压缩；叠加 2/3/4-gram 哈希记忆。'],
    ['Reuse', '复用第 1 层的全局 KV 与已选稀疏位置，减少重复索引计算。'],
    ['Decoder Full', '全局 KV 源自最终编码器状态；自行构造索引，不再压缩分组。'],
    ['Decoder Reuse', '复用第 3 层的 KV 与稀疏位置，保持因果解码。'],
    ['Decoder Reindex', '复用第 3 层的 KV，依据本层状态重新选择稀疏位置。']];
  document.querySelectorAll('[data-layer]').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('[data-layer]').forEach(e => e.classList.toggle('selected', e === b));
    const i = Number(b.dataset.layer); $('layer-info').replaceChildren();
    const title = document.createElement('b'); title.textContent = `Layer 0${i} · ${layerDescriptions[i][0]}`;
    $('layer-info').append(title, document.createElement('br'), layerDescriptions[i][1]);
  }));
  const reset = async () => {
    const count=await rpc('shape',config());
    if(!count) throw new Error('此配置无效或超过 500,000 参数；请选择较小的层数/宽度。');
    const info = await rpc('reset', config()); clearExperiment(info); log('创建全新随机模型；已保存的阶段模型继续保留。');
  };
  action('create-model', reset); action('reset', reset);
  action('train-all', async () => {
    const options = trainingOptions(true);
    if (!Number.isFinite(options.lr) || options.lr < 0 || options.lr > .01) throw new Error('学习率范围应为 0–0.01');
    await reset(); training = true; setChart('live');
    try { await rpc('run', options); } catch (error) { training = false; throw error; }
  });
  action('train-stage', async () => {
    const options = trainingOptions(false); training = true; setChart('live');
    try { await rpc('run', options); } catch (error) { training = false; throw error; }
  });
  action('pause', async () => {
    const s = await rpc('pause'); paused = s.paused; $('pause').textContent = paused ? '继续训练' : '暂停';
    $('train-label').textContent = `${stageName(s)} · ${paused ? '已暂停' : '训练中'}`;
    log(paused ? '已暂停，权重和优化器保持在内存中。' : '继续训练。');
  });
  action('save-model', async () => download('my-model.aiadd', JSON.stringify({ format: 'aiadd-experiment-v1', cot: latest.cot !== false, protocol: latest.protocol, model_base64: pack(await rpc('export', { optimizer: false })) }), 'application/json'));
  action('save-checkpoint', async () => download('my-training-checkpoint.aiadd', JSON.stringify({ format: 'aiadd-experiment-v1', cot: latest.cot !== false, protocol: latest.protocol, model_base64: pack(await rpc('export', { optimizer: true })) }), 'application/json'));
  action('import', async () => $('file-input').click());
  $('file-input').addEventListener('change', () => task(async () => {
    const file = $('file-input').files[0]; if (!file) return;
    if (file.size > 32 * 1024 * 1024) throw new Error('请导入 32 MiB 以内的模型文件');
    let bytes = await file.arrayBuffer(), cot = true, protocol = 'legacy-v1';
    if (!file.name.endsWith('.cn')) { const bundle = JSON.parse(new TextDecoder().decode(bytes)); if (bundle.format !== 'aiadd-experiment-v1') throw new Error('请选择 .cn 或 .aiadd 模型包'); bytes = unpack(bundle.model_base64); cot = bundle.cot !== false; protocol = bundle.protocol || 'legacy-v1'; }
    const info = await rpc('load', { bytes, cot, protocol, student: true });
    clearExperiment(info); $('dim').value = info.dim; $('ff').value = info.ff; $('layers').value = info.layers;
    syncTrainingMode(cot);
    $('train-label').textContent = `已导入模型：${file.name}`;
    log(`导入 ${file.name}；请选择阶段继续训练。`);
    $('file-input').value = '';
  }).catch(() => {}));
  $('addition-form').addEventListener('submit', e => { e.preventDefault(); task(infer).catch(() => {}); });
  $('infer-model').addEventListener('change', () => task(selectModel).catch(() => {}));
  $('builtin-variant').addEventListener('change',()=>task(selectModel).catch(()=>{}));
  $('infer-style').addEventListener('change', clearAnswer);
  $('infer-cache').addEventListener('change', clearAnswer);
  document.querySelectorAll('[data-a]').forEach(b => b.addEventListener('click', () => {
    $('operand-a').value = b.dataset.a; $('operand-b').value = b.dataset.b;
    clearAnswer();
  }));
  ['operand-a', 'operand-b'].forEach(id => $(id).addEventListener('input', clearAnswer));
  $('random-question').addEventListener('click', () => {
    const width = $('question-width').value === 'mixed' ? 1 + Math.floor(Math.random() * 100) : Number($('question-width').value === 'custom' ? $('custom-width').value : $('question-width').value);
    if (!Number.isSafeInteger(width) || width < 1) { toast('请输入正整数位数'); return; }
    const operand = () => Array.from({ length: width }, (_, i) => String(i === 0 && width > 1 ? 1 + Math.floor(Math.random() * 9) : Math.floor(Math.random() * 10))).join('');
    $('operand-a').value = operand(); $('operand-b').value = operand(); clearAnswer();
  });
  $('question-width').addEventListener('change', () => { $('custom-width-field').hidden = $('question-width').value !== 'custom'; });
  $('download-attempt').addEventListener('click', () => { if (lastAttempt) download('addition-attempt.json', JSON.stringify(lastAttempt, null, 2), 'application/json'); });
  window.addEventListener('aiadd:example', e => task(async () => {
    $('operand-a').value = e.detail.a; $('operand-b').value = e.detail.b;
    $('infer-model').value = stageFiles.at(-1);
    $('builtin-variant').value='selected';
    $('infer-style').value = 'auto';
    if (activeName !== stageFiles.at(-1)) { await rpc('load', { name: stageFiles.at(-1), student: false }); activeName = stageFiles.at(-1); }
    location.hash = 'play'; await infer();
  }).catch(() => {}));
  $('stage').addEventListener('change', () => {
    syncTrainingMode($('train-cot').checked);
    window.dispatchEvent(new CustomEvent('aiadd:show-data', { detail: $('stage').value }));
  });
  $('train-protocol').addEventListener('change',()=>{window.dispatchEvent(new CustomEvent('aiadd:protocol',{detail:$('train-protocol').value}));});
  $('train-cot').addEventListener('change', () => syncTrainingMode($('train-cot').checked));
  action('evaluate', async () => {
    $('evaluation-result').textContent = '正在运行所选模式的批量测试，请保持页面打开…';
    const style = $('infer-style').value;
    const r = await rpc('evaluate', { student: isStudent(), ...(style === 'auto' ? {} : { cot: style === 'cot' }) }); latestFullTest = { source: sourceName(), ...r };
    $('evaluation-result').textContent = `${r.passed ? '全部通过' : '存在错误'}：${Object.values(r.tests).map(t => `${t.correct}/${t.total}`).join(' · ')}`;
    download('evaluation.json', JSON.stringify(latestFullTest, null, 2), 'application/json');
  });
  $('chart-recorded').addEventListener('click', () => setChart('recorded'));
  $('chart-structured').addEventListener('click',()=>setChart('structured'));
  $('chart-live').addEventListener('click', () => setChart('live'));
  $('download-page').href = URL.createObjectURL(new Blob([originalDocument], { type: 'text/html;charset=utf-8' }));
  $('download-page').download = 'ai-add.html';
  $('download-builtin').addEventListener('click', () => {
    const variant=sourceName()===stageFiles.at(-1)?$('builtin-variant').value:'selected';
    const name=variant==='selected'?stageFiles.at(-1):stageFiles.at(-1).replace('.cn',`-${variant}.cn`);
    const model=payload.models[name]||payload.variants[name];
    download(name.replace('.cn','.aiadd'),JSON.stringify({format:'aiadd-experiment-v1',protocol:'chat-v2',cot:true,stage:stageFiles.length,variant,model_base64:model.base64}),'application/json');
  });
  $('download-report').addEventListener('click', () => download('ai-add-experiment.json', JSON.stringify({
    recorded: payload.evidence, model_sha256: Object.fromEntries(Object.entries({...payload.models,...payload.variants}).map(([k, v]) => [k, v.sha256])),
    dialogue_training_csv: payload.dialogue_training, dialogue_stages: payload.dialogue_stages,
    current_experiment: { settings: { layers:$('layers').value, dim: $('dim').value, ff: $('ff').value, seed: $('seed').value, cot: $('train-cot').checked, protocol: $('train-protocol').value }, latest, checkpoints: history, logs },
    saved_stages: Array.from(archive.values(), ({ bytes, ...metadata }) => metadata), last_attempt: lastAttempt,
    full_test: latestFullTest
  }, null, 2), 'application/json'));
  window.addEventListener('resize', drawChart); setChart('structured');
  // Small public API for repeatable integration checks and embedding.
  window.aiAdd = { rpc, get initialized() { return ready; }, get ready() { return ready && busy === 0; }, get training() { return training; }, get history() { return history; } };
  init();
})();
