(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const files = JSON.parse($('payload').textContent).datasets;
  const local = files['local.tsv'].trim().split(/\r?\n/).map((line, i) => {
    const [a, b, cin, digit, cout] = line.split('\t').map(Number);
    return { id: i + 1, a, b, cin, digit, cout, line };
  });
  const widths = [1, 2, 4, 6, 10];
  const curriculum = files['curriculum.tsv'].trim().split(/\r?\n/).map((line, i) => {
    const [stage, a, b] = line.split('\t');
    return { id: i + 1, width: widths[Number(stage)], a, b, line };
  });
  const specials = { '<|im_start|>': 256, '<|im_end|>': 257, '<|endoftext|>': 258, '<think>': 259, '</think>': 260 };
  const stageNotes = [null,
    ['200 种可能输入，在线随机采样；每个更新使用 4 条复制序列。', '先学习数字和格式；输出复制 a、b，不包含加法标签。'],
    ['55 条无进位样本：cin = 0 且 a + b < 10。', '只学习不产生进位的简单加法。'],
    ['100 条样本：cin = 0，覆盖全部 a、b。', '加入会产生输出进位的样本。'],
    ['200 条样本：10 × 10 × 2，覆盖全部 a、b、cin。', '训练包括输入进位；错误状态会被额外重采样。'],
    ['2,560 道课程算式：1、2、4、6、10 位各 512 道。每 8 道插入一道连续进位题。', 'RL 实际依次采样 2 位 → 4 位 → 10 位组，其他组保留在课程数据中。每道题采样 4 条轨迹，用奖励差异更新。'],
    ['200 条逐位样本，沿用 SFT 标签；错误或间隔较小的状态会被额外采样。', '权重和激活模拟量化后，继续学习结果位与进位，修复量化误差。']
  ];
  let rows = [], selected = 0, page = 0, cot = true, chat = true, protocol = 'chat-v2', detailSerial = 0;
  const pageSize = 8;
  const stage = () => Number($('data-stage').value);
  function tokenID(token) { return specials[token] ?? token.charCodeAt(0); }
  function tokenDescription(tokens, pretrain) {
    return tokens.map((token, i) => `${String(i).padStart(2)}  ${JSON.stringify(token).padEnd(16)} id=${String(tokenID(token)).padStart(3)}  ${pretrain ? (i ? '预测目标' : '起始上下文') : (i >= 5 ? '监督输出' : '输入上下文')}`).join('\n');
  }
  function select(index) {
    selected = index; page = Math.floor(selected / pageSize); renderList(); renderDetail();
  }
  function renderList() {
    const list = $('sample-list'); list.replaceChildren();
    rows.slice(page * pageSize, (page + 1) * pageSize).forEach((row, offset) => {
      const index = page * pageSize + offset;
      const button = document.createElement('button'); button.className = `sample-row${index === selected ? ' selected' : ''}`;
      button.setAttribute('aria-pressed', String(index === selected));
      const num = document.createElement('small'); num.textContent = `#${row.id}`;
      const text = document.createElement('span');
      text.textContent = protocol === 'column-v2' && cot ? `a=${row.a}  b=${row.b}  cin=${row.cin}` : chat ? stage() === 1 ? `Copy: ${row.a}` : `${row.a} + ${row.b}` : stage() === 5 || (!cot && stage() !== 1) ? `${row.a} + ${row.b}` : `a=${row.a}  b=${row.b}  cin=${row.cin}`;
      button.append(num, text); button.addEventListener('click', () => select(index)); list.append(button);
    });
    $('sample-count').textContent = `${rows.length} 条`;
    $('sample-page').textContent = `${page + 1} / ${Math.ceil(rows.length / pageSize)}`;
    $('sample-prev').disabled = page === 0; $('sample-next').disabled = (page + 1) * pageSize >= rows.length;
  }
  function renderDetail() {
    const row = rows[selected], n = stage();
    if (protocol==='column-v2' && cot) { renderColumn(row,n); return; }
    if (chat) { renderChat(row, n); return; }
    if (!cot && n !== 1) {
      const a = String(row.a).replace(/^0+(?=\d)/, ''), b = String(row.b).replace(/^0+(?=\d)/, ''), answer = (BigInt(a) + BigInt(b)).toString();
      const input = `<|im_start|>${a}+${b}=`, output = `${answer}<|endoftext|>`;
      const tokens = ['<|im_start|>', ...`${a}+${b}=`, ...answer, '<|endoftext|>'];
      $('sample-title').textContent = `无 CoT 样本 · ${a} + ${b}`; $('sample-kind').textContent = n === 5 ? 'DIRECT RLOO' : 'ANSWER ONLY';
      $('sample-purpose').textContent = '无 CoT 实验直接监督最终答案，不包含进位或思考 token。1 位 → 2 位 → 1/2/4/6/10 位课程。';
      $('sample-output-label').textContent = n === 5 ? '独立奖励参照（不是固定采样输出）' : '最终答案目标';
      $('sample-input').textContent = input; $('sample-output').textContent = output;
      $('sample-objective').textContent = n === 5 ? '同题采样 4 个最终答案，完整答案及结束标记正确奖励 1，否则 0；不使用中间步骤奖励。' : '只对最终答案数字与 <|endoftext|> 计算交叉熵；提示词不计入交叉熵。';
      $('sample-record').textContent = '# 在线生成规则预览，不代表某一步实际随机抽样\n' + input + output;
      $('sample-token-title').textContent = '完整直接回答序列 / token ID';
      $('sample-tokens').textContent = tokens.join('') + '\n[' + tokens.map(tokenID).join(', ') + ']';
      $('sample-format-note').textContent = '提示词从整条算式开始；取消勾选会实际切换目标函数和 RL 奖励，不是仅仅隐藏思维链。';
      return;
    }
    $('sample-title').textContent = `样本 #${row.id}${n === 5 ? ` · ${row.width} 位加法` : ` · a=${row.a}, b=${row.b}, cin=${row.cin}`}`;
    $('sample-purpose').textContent = stageNotes[n][1];
    $('sample-kind').textContent = n === 1 ? 'PRETRAIN' : n === 5 ? 'RLOO' : n === 6 ? 'QAT' : 'SFT';
    $('sample-output-label').textContent = n === 5 ? '如何产生输出与奖励' : '训练目标输出';
    if (n === 5) {
      $('sample-input').textContent = `<|im_start|>user\n${row.a}+${row.b}\n<|im_end|>`;
      $('sample-output').textContent = '数据集没有固定答案或固定 CoT。\n当前策略采样 4 条完整逐位轨迹，\n再对轨迹和最终答案计算奖励。';
      $('sample-objective').textContent = 'reward = 0.5 × 正确列比例 + 0.5 × 整题正确。RLOO 比较同题 4 条采样的奖励，保留参考策略 KL 正则。';
      $('sample-record').textContent = `# curriculum.tsv: stage\ta\tb\n${row.line}`;
      const a = row.a.at(-1), b = row.b.at(-1);
      $('sample-token-title').textContent = '第一列的实际输入模板 / 后续列由预测进位驱动';
      $('sample-tokens').textContent = `<|im_start|>${a}${b}0<think>\n输入 token ID: [256, ${a.charCodeAt(0)}, ${b.charCodeAt(0)}, 48, 259]\n→ 采样 digit → 采样 cout → 采样 </think>\n→ 下一列使用这次采样的 cout`;
      $('sample-format-note').textContent = '上方消息是算式展示格式。实际 RL 按列采样 token，没有把整段 user 消息送入网络。';
      const button = document.createElement('button'); button.className = 'btn small'; button.textContent = '让成品回答这条题，查看实际 CoT →';
      button.addEventListener('click', () => window.dispatchEvent(new CustomEvent('aiadd:example', { detail: { a: row.a, b: row.b } })));
      $('sample-format-note').append(document.createElement('br'), button);
    } else {
      const tokens = ['<|im_start|>', String(row.a), String(row.b), String(row.cin), '<think>', String(n === 1 ? row.a : row.digit), String(n === 1 ? row.b : row.cout), '</think>'];
      $('sample-input').textContent = tokens.slice(0, 5).join(''); $('sample-output').textContent = tokens.slice(5).join('');
      $('sample-objective').textContent = n === 1 ? '预训练对序列第 1–7 位做下一 token 预测，包括数字、格式和复制目标。' : `digit=${row.digit}，cout=${row.cout}；只监督最后 3 个 token（结果位、进位、</think>）。输入字段不计入交叉熵目标。`;
      $('sample-record').textContent = n === 1 ? `# 在线生成的复制样本（可能样本预览）\n${tokens.join('')}` : `# local.tsv: a\tb\tcin\tdigit\tcout\n${row.line}`;
      $('sample-token-title').textContent = '完整训练序列 / token ID / 监督位置';
      $('sample-tokens').textContent = `${tokens.join('')}\n[${tokens.map(tokenID).join(', ')}]\n\n${tokenDescription(tokens, n === 1)}`;
      $('sample-format-note').textContent = '原始序列没有字段名、空格或自然语言。a、b、cin 是相邻的单字节数字；<|im_start|>=256，<think>=259，</think>=260。';
    }
  }
  function decode(tokens) {
    const names = {256:'<|im_start|>',257:'<|im_end|>',258:'<|endoftext|>',259:'<think>',260:'</think>'};
    return tokens.map(t => names[t] || String.fromCharCode(t)).join('');
  }
  async function renderChat(row,n) {
    const serial=++detailSerial;
    $('sample-title').textContent=`对话样本 #${row.id} · ${n===1?'数字复制':`${row.a} + ${row.b}`}`;
    $('sample-input').textContent='等待 C 引擎生成真实训练序列…';
    $('sample-output').textContent=''; $('sample-record').textContent=''; $('sample-tokens').textContent='';
    if (!window.aiAdd?.initialized) return;
    try {
      const sample=await window.aiAdd.rpc('example',{a:row.a,b:row.b,cot,copy:n===1,protocol});
      if(serial!==detailSerial||!chat)return;
      if(n===7) {
        const input=decode(sample.tokens.slice(0,sample.prompt));
        $('sample-title').textContent=`自由草稿 RL 提示 · ${row.a} + ${row.b}`;
        $('sample-kind').textContent='OUTCOME-ONLY RLOO';
        $('sample-purpose').textContent='这一阶段没有标准 CoT，也不把示范步骤交给模型。模型连续采样四次，草稿可自由改写。';
        $('sample-input').textContent=input;
        $('sample-output-label').textContent='无固定 assistant 目标';
        $('sample-output').textContent=`由当前策略自由采样${cot?'一个 <think>…</think> 草稿和最终答案':'最终答案（不含思考块）'}，到消息结束或上下文预算耗尽。独立答案参照：${BigInt(row.a)+BigInt(row.b)}。`;
        $('sample-objective').textContent='只检查完整消息、正常思考边界和最终数字答案。正确得 1，否则 0；不检查 pos/a/b/cin 等字段、中间算式或推理过程。四条轨迹做 RLOO + 冻结参考策略 KL；同奖励组跳过更新。选模也使用真实自由生成的最终答案。';
        $('sample-record').textContent=JSON.stringify({messages:[{role:'user',content:`${row.a}+${row.b}`}],expected_answer:String(BigInt(row.a)+BigInt(row.b)),reward:'legal_message && correct_final_answer ? 1 : 0'},null,2);
        $('sample-token-title').textContent='只有提示词是数据；草稿来自真实策略采样';
        $('sample-tokens').textContent=input+'\n['+sample.tokens.slice(0,sample.prompt).join(', ')+']';
        $('sample-format-note').textContent='阶段 7 的实际四份采样与奖励见训练控制台“RL 草稿纸”，包括错误答案、格式失败和均奖未更新的轮次。';
        return;
      }
      const {tokens,prompt}=sample;
      const input=decode(tokens.slice(0,prompt)),output=decode(tokens.slice(prompt));
      $('sample-kind').textContent=n===1?'CHAT PRETRAIN':n===5?'CHAT RLOO':cot?'CHAT + CoT':'CHAT DIRECT';
      $('sample-purpose').textContent=n===1?'学习 user/assistant 对话格式与数字复制，不使用加法标签，也不使用思考标记。':'整道算式放在 user 消息；assistant 连续生成一次完整回答，所有字段与数字均由网络输出。'+(protocol==='chat-v2'?'思考块每行是 pos=位置 a=数字 b=数字 cin=输入进位 -> 结果位 输出进位，从 pos=0 的个位开始；块后给出正序最终答案。':'这是历史紧凑格式检查点的训练预览。');
      $('sample-input').textContent=input;
      $('sample-output-label').textContent=n===5?'奖励参照示例（实际输出由策略采样）':'assistant 监督目标';
      $('sample-output').textContent=output;
      $('sample-objective').textContent=n===5?(cot?'同题采样 4 个完整 assistant 回答，以步骤与最终答案奖励做 RLOO，保留 KL 与熵项。':'同题采样 4 个完整 assistant 回答，答案及对话结束标记正确奖励 1，否则 0。'):'user 消息和 assistant 角色前缀不计交叉熵；监督全部 assistant 内容及结束标记，数字权重 3、其他 token 权重 1，再按每条回答权重归一化。';
      $('sample-record').textContent=JSON.stringify({messages:[{role:'user',content:n===1?`Copy: ${row.a}`:`${row.a}+${row.b}`},{role:'assistant',content:output.replace(/<\|im_end\|><\|endoftext\|>$/,'')}]},null,2);
      $('sample-token-title').textContent='由 C 训练器生成的完整对话 / token ID / 监督位置';
      $('sample-tokens').textContent=decode(tokens)+'\n['+tokens.join(', ')+']\n\n'+tokens.map((id,i)=>`${String(i).padStart(3)} ${JSON.stringify(decode([id])).padEnd(16)} id=${id} ${i<prompt?'提示词（不计损失）':'assistant 监督'}`).join('\n');
      $('sample-format-note').textContent='预览直接调用 C 训练器同一个样本函数。每段 CoT 回答只有一个 <think>…</think>，包住全部步骤；关闭 CoT 或预训练均不包含思考标记。这里是在线课程的可能样本，不代表某一步实际抽样。';
    } catch(error) { if(serial===detailSerial)$('sample-input').textContent=error.message; }
  }
  async function renderColumn(row,n) {
    const serial=++detailSerial;
    if(!window.aiAdd?.initialized)return;
    try {
      const sample=await window.aiAdd.rpc('columnExample',{a:row.a,b:row.b,cin:row.cin||0,copy:n===1});
      if(serial!==detailSerial)return;
      const input=decode(sample.tokens.slice(0,sample.prompt)),output=decode(sample.tokens.slice(sample.prompt));
      $('sample-title').textContent=`逐位对话 #${row.id} · a=${row.a} b=${row.b} cin=${row.cin||0}`;
      $('sample-kind').textContent=n===1?'COLUMN COPY':n===5?'COLUMN RLOO':'COLUMN SFT';
      $('sample-purpose').textContent='这是实际送入网络的完整角色消息。pos 只记录在整题轨迹中，运算规则不依赖位置。';
      $('sample-input').textContent=input;$('sample-output').textContent=output;
      $('sample-output-label').textContent=n===5?'奖励参照（实际四组轨迹由策略采样）':'assistant 监督目标';
      $('sample-objective').textContent=n===1?'只监督 assistant 复制 a、空格、b 和消息结束标记，不使用加法标签。':'只监督 assistant 的结果位、空格、进位和两种消息结束标记。局部调用不用 <think> 或 </think> 充当结束标记。';
      $('sample-record').textContent=JSON.stringify({messages:[{role:'user',content:`${n===1?'copy':'add'} a=${row.a} b=${row.b} cin=${row.cin||0}`},{role:'assistant',content:output.replace(/<\|im_end\|><\|endoftext\|>$/,'')}]},null,2);
      $('sample-token-title').textContent='C 训练器真实逐位消息 / token 与监督位置';
      $('sample-tokens').textContent=decode(sample.tokens)+'\n['+sample.tokens.join(', ')+']\n\n'+sample.tokens.map((id,i)=>`${i} ${JSON.stringify(decode([id]))} id=${id} ${i<sample.prompt?'提示词（不计损失）':'assistant 监督'}`).join('\n');
      $('sample-format-note').textContent='整题调度器只取位、传递模型预测进位、拼接预测数字；不计算结果位。每步 user/assistant 原文均在推理区保留，整个逐位轨迹只展示一个思考块。';
    } catch(e){$('sample-format-note').textContent=e.message;}
  }
  function refresh() {
    if(protocol!=='chat-v2' && stage()===7) $('data-stage').value=6;
    const n = stage();
    if(protocol==='column-v2' && cot){
      const labels=['复制与角色格式','一位无进位','学习输出进位','学习输入进位','2/4/10 位完整轨迹 RLOO','逐位 FP4 QAT'];
      Array.from($('data-stage').options).forEach((o,i)=>{o.textContent=`0${i+1} · ${labels[i]||'仅新对话支持'}`;o.disabled=i===6&&protocol!=='chat-v2';});
      $('data-width-field').hidden=true;
      rows=local.filter(r=>n===2?!r.cin&&r.a+r.b<10:n===3?!r.cin:true);
      $('data-summary').textContent=`${rows.length} 种实际逐位对话状态 · 规范 user/assistant · 无误用思考结束标记`;
      $('data-rule').textContent=n===5?'RLOO 逐次采样 2/4/10 位完整轨迹，每列接收上列模型预测的进位。下方展示可能的逐位调用。':n===1?'在线采样 a/b/cin，assistant 复制 a 和 b，不使用加法标签。':n===2?'采样 cin=0 且 a+b<10 的状态，学习无进位加法；下方列出全部可能状态。':'本阶段按状态均匀采样；监督预测结果位与进位。位置编号属于调度元数据，不进入模型。';
      select(0);return;
    }
    if(chat){
      const labels=['数字复制与对话格式','一位无进位','两位算式','1/2/4/6/10 位课程','完整对话 RLOO','完整对话 QAT','自由草稿 RL · 只看最终答案'];
      Array.from($('data-stage').options).forEach((o,i)=>{o.textContent=`0${i+1} · ${labels[i]||'仅新对话支持'}`;o.disabled=i===6&&protocol!=='chat-v2';});
      $('data-width-field').hidden=n===2;
      Array.from($('data-width').options).forEach(option=>option.disabled=n===3&&Number(option.value)>2);
      if(n===3&&Number($('data-width').value)>2)$('data-width').value='all';
      rows=(n===2?local.filter(r=>!r.cin&&r.a+r.b<10):curriculum.filter(r=>(n!==3||r.width<=2)&&($('data-width').value==='all'||r.width===Number($('data-width').value)))).map(r=>({...r,a:String(r.a).replace(/^0+(?=\d)/,''),b:String(r.b).replace(/^0+(?=\d)/,'')}));
      $('data-summary').textContent=`完整 user / assistant 对话 · ${n===1?'无思考标记的复制预训练':cot?'一个思考块 + 最终答案':'直接生成最终答案'}`;
      $('data-rule').textContent='按在线课程展示可能样本，token 由 C 训练器生成。两位 SFT 混合一位题；长课程在前半程解锁五档，后半程混合全部长度；RL/QAT 从开始混合全部位数。';
      select(0); return;
    }
    const legacyLabels = ['旧协议预训练', '旧协议无进位', '旧协议输出进位', '旧协议输入进位', '旧协议 RLOO', '旧协议 QAT'];
    Array.from($('data-stage').options).forEach((o, i) => o.textContent = `0${i + 1} · ${legacyLabels[i]}`);
    $('data-width-field').hidden = n !== 5;
    rows = !cot && n !== 1 ? (n === 2 ? local.filter(r => !r.cin && r.a + r.b < 10) : curriculum.filter(r => n === 3 ? r.width === 2 : n === 5 ? $('data-width').value === 'all' || r.width === Number($('data-width').value) : true)) : n === 5 ? curriculum.filter(r => $('data-width').value === 'all' || r.width === Number($('data-width').value)) : local.filter(r => n === 2 ? !r.cin && r.a + r.b < 10 : n === 3 ? !r.cin : true);
    $('data-summary').textContent = stageNotes[n][0];
    $('data-rule').textContent = n === 5 ? '原始 curriculum.tsv，保留前导零；0/1/2/3/4 对应 1/2/4/6/10 位。' : n === 1 ? '按训练器的在线生成规则枚举；这里不代表某一步实际抽到的顺序。' : '直接读取随网页内置的 local.tsv，按本阶段课程规则筛选。';
    if (!cot && n !== 1) { $('data-summary').textContent = '无 CoT：整条算式 → 最终答案。这里按实际在线课程规则展示可能样本。'; $('data-rule').textContent = '使用预设算式展示在线生成的目标格式；真实训练按随机种子重新采样，不读取这些预览行。'; }
    select(cot && (n === 4 || n === 6) ? 159 : 0);
  }
  $('data-stage').addEventListener('change', refresh); $('data-width').addEventListener('change', refresh);
  $('sample-random').addEventListener('click', () => select(Math.floor(Math.random() * rows.length)));
  $('sample-prev').addEventListener('click', () => select((page - 1) * pageSize));
  $('sample-next').addEventListener('click', () => select((page + 1) * pageSize));
  $('dataset-download').addEventListener('click', () => {
    if(chat)return;
    const n = stage();
    const content = !cot && n !== 1 ? rows.map(r => `<|im_start|>${r.a}+${r.b}=${BigInt(r.a) + BigInt(r.b)}<|endoftext|>`).join('\n') : n === 1 ? rows.map(r => `<|im_start|>${r.a}${r.b}${r.cin}<think>${r.a}${r.b}</think>`).join('\n') + '\n' : rows.map(r => r.line).join('\n') + '\n';
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([content], { type: 'text/plain;charset=utf-8' })); a.download = !cot && n !== 1 ? `direct-stage-${n}-possible-samples.txt` : n === 1 ? 'pretrain-possible-samples.txt' : n === 5 ? `curriculum-${$('data-width').value}.tsv` : `stage-${n}-local.tsv`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('dataset-download').addEventListener('click',async()=>{
    if(!chat)return;
    try{
      const col=protocol==='column-v2'&&cot;
      const samples=stage()===7?null:await window.aiAdd.rpc(col?'columnExamples':'examples',{rows,cot,copy:stage()===1,protocol});
      const content=stage()===7?rows.map(r=>JSON.stringify({messages:[{role:'user',content:`${r.a}+${r.b}`}],expected_answer:String(BigInt(r.a)+BigInt(r.b)),reward:'legal_message && correct_final_answer ? 1 : 0'})).join('\n')+'\n':samples.map((s,i)=>JSON.stringify({messages:[{role:'user',content:col?`${stage()===1?'copy':'add'} a=${rows[i].a} b=${rows[i].b} cin=${rows[i].cin||0}`:stage()===1?`Copy: ${rows[i].a}`:`${rows[i].a}+${rows[i].b}`},{role:'assistant',content:decode(s.tokens.slice(s.prompt)).replace(/<\|im_end\|><\|endoftext\|>$/,'')}]})).join('\n')+'\n';
      const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type:'application/x-ndjson'}));a.download=`${col?'column-v2':'chat'}-stage-${stage()}-${cot?'cot':'direct'}.jsonl`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    }catch(error){$('sample-format-note').textContent=error.message;}
  });
  window.addEventListener('aiadd:show-data', e => { $('data-stage').value = e.detail; refresh(); });
  window.addEventListener('aiadd:train-mode', e => { cot = e.detail; refresh(); });
  window.addEventListener('aiadd:protocol', e => { protocol=e.detail;chat=protocol==='chat-v2'||protocol==='chat-v1'||protocol==='column-v2'; refresh(); });
  window.addEventListener('aiadd:ready',refresh);
  refresh();
})();
