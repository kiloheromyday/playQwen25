const $ = (id) => document.getElementById(id);
const state = { messages: [], busy: false, greedy: false, info: null, evaluation: null, training: null, view: 'chat' };
const number = (n, digits=2) => n == null ? '—' : Number(n).toFixed(digits);
const percent = (n) => n == null ? '—' : `${number(n * 100, 1)}%`;
const icons = () => window.lucide?.createIcons();
const element = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text != null) node.textContent = text; return node; };
async function api(url, data) {
  const options = data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)};
  const response = await fetch(url, options);
  if (!response.ok) { const error = await response.json().catch(()=>({})); throw new Error(typeof error.detail === 'string' ? error.detail : `请求失败 (${response.status})`); }
  return response.json();
}
function showError(id, error) { $(id).textContent = error?.message || ''; $(id).hidden = !error; }
function setBusy(busy) { state.busy = busy; $('send-button').disabled=busy; $('run-eval').disabled=busy; $('clear-chat').disabled=busy; }
function switchView(view) {
  state.view=view;
  document.querySelectorAll('.view').forEach(node=>node.classList.toggle('active',node.id===`view-${view}`));
  document.querySelectorAll('.nav-item').forEach(node=>node.classList.toggle('active',node.dataset.view===view));
  $('page-title').textContent={chat:'对话',evaluation:'模型评测',training:'训练记录'}[view];
  if(view==='training') requestAnimationFrame(drawTrainingChart);
}
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>switchView(button.dataset.view));
document.querySelectorAll('[data-prompt]').forEach(button=>button.onclick=()=>{ $('prompt').value=button.dataset.prompt; $('prompt').dispatchEvent(new Event('input')); $('prompt').focus(); });
$('prompt').oninput=()=>{$('input-count').textContent=`${Array.from($('prompt').value).length} 字符`;};
for(const [id,output] of [['temperature','temperature-value'],['top-p','top-p-value'],['penalty','penalty-value']]) $(id).oninput=()=>$(output).textContent=number($(id).value);
function sampling(greedy) {state.greedy=greedy; $('greedy').classList.toggle('selected',greedy); $('sampling').classList.toggle('selected',!greedy); $('temperature').disabled=greedy; $('top-p').disabled=greedy;}
$('greedy').onclick=()=>sampling(true); $('sampling').onclick=()=>sampling(false);
$('reset-settings').onclick=()=>{ sampling(false); for(const [id,value] of [['temperature',.7],['top-p',.9],['penalty',1.1],['max-tokens',48],['seed',0]]){$(id).value=value; $(id).dispatchEvent(new Event('input'));} $('use-history').checked=true;};
function appendMessage(role,text,pending=false) {
  $('chat-empty')?.remove();
  const row=element('article',`message ${role}${pending?' pending':''}`);
  row.append(element('span','avatar',role==='user'?'你':'Q'));
  const body=element('div','message-body'), label=element('div','message-label'); label.append(element('strong','',role==='user'?'你':'本地模型'));
  if(role==='assistant'&&!pending){const copy=element('button','copy-button');copy.type='button';copy.title='复制回复';copy.setAttribute('aria-label','复制回复');copy.innerHTML='<i data-lucide="copy"></i>';copy.onclick=async()=>{try{await navigator.clipboard.writeText(text);copy.title='已复制';}catch{copy.title='复制失败';}}; label.append(copy);}
  body.append(label,element('div','message-text',text));row.append(body);$('messages').append(row);icons();$('messages').scrollTop=$('messages').scrollHeight;return row;
}
$('clear-chat').onclick=()=>{state.messages=[];$('messages').replaceChildren();showError('chat-error',null);};
$('chat-form').onsubmit=async(event)=>{
  event.preventDefault(); if(state.busy) return;
  const text=$('prompt').value.trim();if(!text)return;
  const history=[...state.messages,{role:'user',content:text}];appendMessage('user',text);$('prompt').value='';$('prompt').dispatchEvent(new Event('input'));
  const pending=appendMessage('assistant','正在生成…',true);setBusy(true);showError('chat-error',null);
  try{
    const result=await api('/api/chat',{messages:history.slice(-49),history:$('use-history').checked,max_new_tokens:Number($('max-tokens').value),temperature:state.greedy?0:Number($('temperature').value),top_p:Number($('top-p').value),repetition_penalty:Number($('penalty').value),seed:Number($('seed').value)});
    pending.remove();const row=appendMessage('assistant',result.text||'（本次生成了结束标记，没有可显示文本）');
    const meta=element('div','message-meta');for(const text of [`${result.generated_tokens} token`,`${number(result.latency_ms,0)} ms`,`${number(result.tokens_per_second,1)} token/s`,result.stopped_on_eos?'EOS 结束':'达到生成上限'])meta.append(element('span','',text));row.querySelector('.message-body').append(meta);
    const values=[`${number(result.ttft_ms,1)} ms`,`${number(result.tokens_per_second,1)} token/s`,percent(result.unknown_input_rate),`${result.truncated_tokens} token`];
    $('last-metrics').querySelectorAll('dd').forEach((node,i)=>node.textContent=values[i]);
    state.messages=[...history,{role:'assistant',content:result.text}].slice(-48);$('messages').scrollTop=$('messages').scrollHeight;
  }catch(error){pending.remove();showError('chat-error',error);$('prompt').value=text;$('prompt').dispatchEvent(new Event('input'));}finally{setBusy(false);}
};
$('run-eval').onclick=async()=>{
  if(state.busy)return;setBusy(true);showError('eval-error',null);$('eval-progress').hidden=false;$('eval-state').textContent='运行中';
  try{const report=await api('/api/evaluate',{custom_text:$('custom-text').value});renderEvaluation(report);}catch(error){showError('eval-error',error);$('eval-state').textContent='评测失败';}finally{$('eval-progress').hidden=true;setBusy(false);}
};
$('export-eval').onclick=()=>{location.href='/api/evaluation/download';};
function metric(label,value,caption){const node=element('div','metric');node.append(element('div','metric-label',label),element('div','metric-number',value),element('div','metric-caption',caption));return node;}
function renderEvaluation(report){
  if(!report)return;state.evaluation=report;$('eval-empty').hidden=true;$('eval-results').hidden=false;$('export-eval').disabled=false;$('eval-state').textContent=`已完成 · ${number(report.elapsed_seconds,1)} s`;
  const v=report.trained.validation||report.trained.train, baseline=report.random_baseline.validation||report.random_baseline.train;
  $('metric-grid').replaceChildren(metric('验证集 Loss ↓',number(v.loss,3),`随机基线 ${number(baseline.loss,3)}`),metric('验证集 PPL ↓',number(v.ppl,1),'同一词表下比较'),metric('下一个 token · Top-1 ↑',percent(v.top1_accuracy),`Top-5 ${percent(v.top5_accuracy)}`),metric('验证输入 · 未知字符 ↓',percent(v.unknown_input_rate),`${v.target_tokens} 个预测目标`));
  $('score-table').replaceChildren();const names={train:'训练集',validation:'验证集',custom:'独立文本'};
  for(const [key,score] of Object.entries(report.trained)) for(const [name,s,className] of [[`${names[key]} · 已训练`,score,''],['随机初始化',report.random_baseline[key],'baseline']]){const row=element('tr',className);for(const value of [name,number(s.loss,3),number(s.ppl,1),percent(s.top1_accuracy),percent(s.top5_accuracy),percent(s.unknown_input_rate),s.target_tokens]) row.append(element('td','',value));$('score-table').append(row);}
  $('probe-results').replaceChildren();for(const probe of report.probes){const node=element('article','probe'),head=element('div','probe-head');head.append(element('strong','',probe.source),element('span','tag',probe.exact_match?'与参考完全一致':'与参考不同'));node.append(head);const lines=element('dl','probe-lines');for(const [label,value] of [['前缀',probe.prompt],['生成',probe.text||'（结束标记）'],['参考',probe.reference]]) lines.append(element('dt','',label),element('dd','',value));node.append(lines);const stats=element('div','probe-stats');for(const value of [`首 token ${number(probe.ttft_ms,1)} ms`,`${number(probe.tokens_per_second,1)} token/s`,`重复二元组 ${percent(probe.repeated_bigram_rate)}`,`峰值分配 ${number(probe.peak_allocated_mib,1)} MiB`])stats.append(element('span','',value));node.append(stats);$('probe-results').append(node);}icons();
}
function renderTraining(report){
  const root=$('training-content');root.replaceChildren();if(!report){root.append(element('p','muted','当前 checkpoint 目录中没有训练记录。'));return;}state.training=report;
  const best=report.history.reduce((a,b)=>a.valid_loss<b.valid_loss?a:b),grid=element('div','metric-grid');grid.append(metric('训练更新',report.steps,`${report.parameters.toLocaleString()} 参数`),metric('最终训练 Loss',number(report.history.at(-1).train_loss,3),`初始 ${number(report.history[0].train_loss,3)}`),metric('最低验证 Loss',number(best.valid_loss,3),`采样记录中的第 ${best.step} 步`),metric('累计预测目标',report.target_tokens_processed.toLocaleString(),'包含重复读取的训练样本'));root.append(grid);
  const heading=element('div','section-title');heading.append(element('h2','','Loss 曲线'),element('span','','训练集与验证集'));root.append(heading);const plot=element('div','training-chart'),canvas=element('canvas');canvas.id='training-canvas';canvas.setAttribute('role','img');canvas.setAttribute('aria-label','训练与验证损失随参数更新次数的变化');plot.append(canvas);root.append(plot);const key=element('div','chart-key');key.append(element('span','','训练集'),element('span','','验证集'));root.append(key);
  root.append(element('p','muted','横轴：参数更新次数。纵轴：下一个 token 交叉熵 loss，越低越好；每个采样点对完整数据集重新评估。'));
  const finding=element('div','training-finding');finding.append(element('strong','',`第 ${best.step} 步后出现过拟合迹象`),element('p','','训练 loss 持续下降，验证 loss 回升。当前加载的是最终 checkpoint；训练文本太少，继续增加更新次数未必改善泛化。'));root.append(finding);
  const tableWrap=element('div','table-scroll'),table=element('table');table.innerHTML='<thead><tr><th>参数更新次数</th><th>训练 Loss</th><th>验证 Loss</th><th>验证 PPL</th></tr></thead>';const body=element('tbody');for(const row of report.history){const tr=element('tr');for(const value of [row.step,number(row.train_loss,4),number(row.valid_loss,4),number(row.valid_ppl,2)])tr.append(element('td','',value));body.append(tr);}table.append(body);tableWrap.append(table);root.append(tableWrap);requestAnimationFrame(drawTrainingChart);
}
function drawTrainingChart(){
  const canvas=$('training-canvas');if(!canvas||!state.training||state.view!=='training')return;
  const rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,p={l:44,r:20,t:24,b:40},rows=state.training.history,maxX=rows.at(-1).step,maxY=Math.ceil(Math.max(...rows.flatMap(r=>[r.train_loss,r.valid_loss])));ctx.font='11px "Segoe UI",sans-serif';
  const x=s=>p.l+s/maxX*(w-p.l-p.r),y=v=>h-p.b-v/maxY*(h-p.t-p.b);
  ctx.lineWidth=1;for(let v=0;v<=maxY;v++){ctx.strokeStyle='#e8eeea';ctx.beginPath();ctx.moveTo(p.l,y(v));ctx.lineTo(w-p.r,y(v));ctx.stroke();ctx.fillStyle='#829087';ctx.textAlign='right';ctx.fillText(v,p.l-10,y(v)+4);}for(let s=0;s<=maxX;s+=20){ctx.fillStyle='#829087';ctx.textAlign='center';ctx.fillText(s,x(s),h-p.b+22);}
  for(const [field,color] of [['train_loss','#176b54'],['valid_loss','#b3455e']]){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((row,i)=>i?ctx.lineTo(x(row.step),y(row[field])):ctx.moveTo(x(row.step),y(row[field])));ctx.stroke();ctx.fillStyle=color;rows.forEach(row=>{ctx.beginPath();ctx.arc(x(row.step),y(row[field]),3,0,Math.PI*2);ctx.fill();});}ctx.fillStyle='#829087';ctx.textAlign='left';ctx.fillText('loss',10,17);ctx.textAlign='right';ctx.fillText('step',w-8,h-7);
}
window.addEventListener('resize',drawTrainingChart);
async function initialize(){
  try{state.info=await api('/api/info');const info=state.info;$('side-params').textContent=info.parameters.toLocaleString();$('side-step').textContent=info.step;$('side-context').textContent=`${info.config.max_position_embeddings} token`;$('connection').textContent='模型已就绪';$('device-name').textContent=info.gpu.replace('NVIDIA GeForce ','');$('footer-model').textContent=`${info.checkpoint_directory} / step ${info.step}`;$('max-tokens').max=Math.min(96,info.config.max_position_embeddings-1);const [evaluation,training]=await Promise.all([api('/api/evaluation'),api('/api/training')]);renderEvaluation(evaluation);renderTraining(training);}catch(error){$('connection').textContent='连接失败';showError('chat-error',error);}icons();
}
initialize();
