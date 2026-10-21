"""Browser-independent regression checks for the small UI's safety boundaries.

Node's DOM stub checks fetch races and read-only state, not visual browser layout.
Actual layout acceptance is recorded separately by B08's browser inspection.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "src/guandan_agent/web/static/app.js"
NODE = shutil.which("node")

HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Node {
  constructor(tag = '') { this.tagName = tag; this.childNodes = []; this.className = ''; this._text = ''; this.value = ''; this.hidden = false; this.disabled = false; this.listeners = {}; this.attrs = {}; }
  get textContent() { return this._text + this.childNodes.map(node => node.textContent).join(''); }
  set textContent(value) { this._text = String(value); this.childNodes = []; }
  append(...nodes) { for (const node of nodes) { if (node.tagName === '#fragment') this.childNodes.push(...node.childNodes); else this.childNodes.push(node); } }
  replaceChildren(...nodes) { this._text = ''; this.childNodes = []; this.append(...nodes); }
  get firstElementChild() { return this.childNodes[0]; }
  get classList() { const node = this; return { add(value) { node.className += ` ${value}`; }, remove(value) { node.className = node.className.split(' ').filter(part => part !== value).join(' '); }, toggle(value, force) { if (force) this.add(value); else this.remove(value); } }; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  querySelectorAll(tag) { return this.childNodes.flatMap(node => [...(node.tagName === tag ? [node] : []), ...node.querySelectorAll(tag)]); }
}
const ids = new Map();
const document = {
  getElementById(id) { if (!ids.has(id)) ids.set(id, new Node('div')); return ids.get(id); },
  createElement(tag) { return new Node(tag); },
  createTextNode(text) { const node = new Node('#text'); node.textContent = text; return node; },
  createDocumentFragment() { return new Node('#fragment'); },
};
const response = (data, ok = true, status = 200, headers = {}) => ({ ok, status, json: async () => data, headers: { get: key => headers[key] || null } });
const context = vm.createContext({ document, console, assert, response, URL, Date, setInterval: () => 0, localStorage: { getItem: () => null, setItem() {}, removeItem() {} }, fetch: async () => response({ items: [] }) });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const fixtures = `
const baseSession = {session_id:'seat-a',seat:0,match_id:'match-a'};
const baseRun = {run_id:'run-a',status:'completed',read_only:true,log_complete:true,active_match_id:'match-a',sessions:[baseSession]};
const state = (version,session='seat-a') => ({run_id:'run-a',session_id:session,view_version:version,round_id:1,status:'completed',log_complete:true,state:{stage:'play',level_rank:'2',self_hand:['S3','H4'],remaining_counts:[2,3,4,5],leader_seat:null,leading_action:null,facts_status:'trusted'}});
ui.run=baseRun;ui.session=baseSession;ui.view=null;ui.runEpoch=1;ui.sessionEpoch=1;
`;
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  vm.runInContext(fixtures, context);
  await vm.runInContext(process.argv[2], context);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.skipif(NODE is None, reason="Node is needed for browser-independent UI checks")
@pytest.mark.parametrize("scenario", [
    pytest.param(r"""
    (async () => {
      ui.view=state(3); fetch=async(path)=>response(path.endsWith('/state')?state(2):baseRun); await pollState(); assert.equal(ui.view.view_version,3);
      fetch=async()=>{throw new Error('network down')};await pollState();assert.ok($('sync-label').textContent.includes('同步失败'));
      fetch=async(path)=>response(path.endsWith('/state')?state(3):baseRun); await pollState(); assert.equal(ui.view.view_version,3);
      assert.equal($('sync-label').textContent.includes('同步失败'),false);
      fetch=async(path)=>response(path.endsWith('/state')?state(4):baseRun); await pollState(); assert.equal(ui.view.view_version,4);
      let finish; fetch=(path)=>path.endsWith('/state')?new Promise(resolve=>finish=resolve):Promise.resolve(response(baseRun)); const pending=pollState();
      while(!finish) await Promise.resolve();
      ui.sessionEpoch++;ui.session={session_id:'seat-b',seat:1};ui.view=null;
      finish(response(state(5)));await pending;assert.equal(ui.view,null);
      baseRun.sessions.push(ui.session);fetch=async(path)=>response(path.endsWith('/state')?state(1,'seat-b'):baseRun);await pollState();assert.equal(ui.view.view_version,1);
    })()
    """, id="state-version-and-selection-race"),
    pytest.param(r"""
    (async () => {
      const report={run_id:'run-a'};ui.report=report;
      const item={evidence_available:true,session_id:'seat-b',request_id:'seat-b:q1'};
      const bad='<img src=x onerror=alert(1)>';
      const detail={state:{...state(1).state,binding:{seat:1,round_id:1}},calls:[],candidates:null,execution:null,review:null,proposal:{reason:bad,action_id:'L0000',source:'main'},log_complete:true};
      fetch=async()=>response(detail);await openEvidence(report,item);
      assert.equal(ui.session.session_id,'seat-a');assert.equal(ui.session.seat,0);
      assert.ok($('report-decision-detail').textContent.includes(bad));
      assert.equal($('report-decision-detail').querySelectorAll('img').length,0);
      assert.ok($('report-decision-detail').textContent.includes('座位 1'));
      let calls=0;fetch=async()=>{calls++;return response(detail)};
      await openEvidence(report,{...item,evidence_available:false});assert.equal(calls,0);
      ui.configs=[{config_name:'local'}];ui.runs=[baseRun];updateControls();
      assert.equal($('start-run').disabled,true);assert.equal($('stop-run').disabled,true);
    })()
    """, id="report-evidence-keeps-seat-and-renders-text"),
    pytest.param(r"""
    (() => {
      const bad='<img src=x onerror=alert(1)>';
      const item={request_id:'seat-a:q1',session_id:'seat-a',evidence_event_ids:['event-a'],evidence_available:true,at_time_facts:[bad],later_facts:['既有公开结果'],assessment:'原有模型判断',suggestion:'仍需评测'};
      const report={kind:'review',report_id:'fixed-report',run_id:'run-a',status:'completed',created_at:'2026-10-19T00:00:00Z',items:[item],errors:[]};
      renderReport(report);
      assert.ok($('report-body').textContent.includes('当时事实 · 模型归纳 · 未核验'));
      assert.ok($('report-body').textContent.includes('事后事实 · 模型归纳 · 未核验 · 当时不可知'));
      assert.ok($('report-body').textContent.includes(bad));
      assert.equal($('report-body').querySelectorAll('img').length,0);
      const modern={...report,generation:{prompt_version:'b09-offline-v2'},items:[{...item,facts_source:'program',facts_verified:true}]};
      renderReport(modern);
      assert.ok($('report-body').textContent.includes('当时事实 · 程序提取 · 已与源记录核对'));
      assert.ok($('report-body').textContent.includes('模型分析 · 仍可能出错'));
      assert.ok($('report-body').textContent.includes('模型建议 · 待验证'));
      assert.ok($('report-body').textContent.includes('事实核验不代表策略有效'));
      renderReport({...modern,generation:{prompt_version:'b09-offline-v3'}});
      assert.ok($('report-body').textContent.includes('当时事实 · 程序提取 · 已与源记录核对'));
      for(const value of [
        {...modern,items:[{...item,facts_source:'program',facts_verified:false}]},
        {...modern,generation:{prompt_version:'b09-offline-v1'}},
        {...modern,generation:{prompt_version:'unknown-version'}},
      ]) {
        renderReport(value);
        assert.ok($('report-body').textContent.includes('声明为程序提取 · 未通过源记录核验'));
        assert.ok($('report-body').textContent.includes('不能当作已核验事实'));
        assert.equal($('report-body').textContent.includes('已与源记录核对'),false);
      }
      renderReport({...modern,items:[{...item,facts_source:'model',facts_verified:true}]});
      assert.ok($('report-body').textContent.includes('当时事实 · 模型归纳 · 未核验'));
      assert.equal($('report-body').textContent.includes('已与源记录核对'),false);
    })()
    """, id="report-fact-provenance-and-model-judgment-labels"),
    pytest.param(r"""
    (async () => {
      fetch=async()=>response({error:{code:'HISTORY_UNAVAILABLE',safe_message:'历史已丢失'}},false,410);
      await loadDecisions(true);assert.equal($('recover-history').hidden,true);
      assert.ok($('decision-list').textContent.includes('较早记录暂不可查询'));
      let requests=0;
      fetch=async()=>response({error:{code:'HISTORY_UNAVAILABLE',safe_message:'历史已丢失'}},false,410,{'X-Retained-From-Seq':String(8 + requests++)});
      await loadDecisions(true);assert.equal(requests,3);assert.equal($('recover-history').hidden,false);assert.equal(ui.retainedFrom,10);
      let path;fetch=async(url)=>{path=url;return response({items:[],has_more:false,next_after_seq:null,retained_from_seq:8,log_complete:false})};
      await $('recover-history').listeners.click();assert.ok(path.includes('after_seq=9'));
      assert.equal($('history-message').hidden,false);
      requests=0;const paths=[];
      fetch=async(url)=>{paths.push(url);return requests++ === 0
        ?response({error:{code:'HISTORY_UNAVAILABLE',safe_message:'记录窗口移动'}},false,410,{'X-Retained-From-Seq':'30'})
        :response({items:[{request_id:'seat-a:q1',opened_seq:32,round_id:1}],has_more:false,next_after_seq:32,retained_from_seq:30,log_complete:true})};
      await loadDecisions(true);assert.equal(requests,2);assert.ok(paths[1].includes('after_seq=29'));
      assert.equal(ui.decisions.length,1);assert.equal(ui.historyStatus,'ready');
      assert.ok($('history-message').textContent.includes('覆盖存在缺口'));
      requests=0;fetch=async()=>{requests++;return response({error:{safe_message:'窗口未进展'}},false,410,{'X-Retained-From-Seq':'1'})};
      await loadDecisions(true);assert.equal(requests,1);assert.equal($('recover-history').hidden,true);
    })()
    """, id="history-gap-needs-server-cursor"),
    pytest.param(r"""
    (async () => {
      ui.mode='reports';ui.run={...baseRun,status:'running',read_only:false};ui.runs=[ui.run];ui.configs=[{config_name:'local'}];
      ui.observedActiveMatchId='match-a';ui.session={session_id:'old-seat-2',seat:2,match_id:'match-a'};
      const newSession={session_id:'new-seat-2',seat:2,match_id:'match-b'};
      let fresh={...ui.run,status:'running',active_match_id:'match-b',sessions:[ui.session,{session_id:'new-seat-0',seat:0,match_id:'match-b'}]};
      let stateFetches=0;
      fetch=async(path)=>{if(path.endsWith('/state')) stateFetches++;return response(path.includes('/decisions?')?{items:[],has_more:false}:fresh)};
      await pollState();assert.equal(ui.session.session_id,'old-seat-2');assert.equal(ui.session.seat,2);
      fresh={...fresh,status:'completed',sessions:[...fresh.sessions,newSession]};
      await pollState();assert.equal(ui.run.status,'completed');assert.equal(ui.session.session_id,'new-seat-2');assert.equal(ui.session.seat,2);
      assert.equal(stateFetches,0);assert.equal($('start-run').disabled,false);assert.equal($('stop-run').disabled,true);
      assert.equal(ui.runs[0].status,'completed');assert.equal(ui.runs[0].read_only,false);assert.equal(ui.runs[0].sessions.length,3);
      selectSession('old-seat-2',{manual:true});assert.equal(ui.followActiveMatch,false);
      while(ui.pollBusy) await Promise.resolve();await pollState();assert.equal(ui.session.session_id,'old-seat-2');
      selectSession('new-seat-2',{manual:true});assert.equal(ui.followActiveMatch,true);
      while(ui.pollBusy) await Promise.resolve();await pollState();assert.equal(ui.session.session_id,'new-seat-2');
      const detail={state:{...state(1).state,binding:{seat:2}},calls:[],candidates:{candidates:[{action_id:'L0001',features:{played_count:2},keep_reasons:[]}]},candidate_actions:[{action_id:'L0001',wire_action:['Pair','3',['S3','H3']]}],execution:null,log_complete:true};
      const display=decisionView(detail,{sessionId:newSession.session_id,requestId:'new-seat-2:q1'});
      assert.ok(display.textContent.includes('对子 · 3'));assert.equal(display.querySelectorAll('span').some(node=>node.attrs['aria-label']==='♥3'),true);
      await selectRun('');assert.equal(ui.run,null);assert.equal($('start-run').disabled,false);
    })()
    """, id="run-summary-refreshes-in-reports-and-keeps-seat"),
    pytest.param(r"""
    (async () => {
      let finish;fetch=()=>new Promise(resolve=>finish=resolve);
      const pending=loadDecisions(true);
      assert.ok($('decision-list').textContent.includes('正在读取决策记录'));
      finish(response({error:{safe_message:'连接中断'}},false,503));await pending;
      assert.ok($('decision-list').textContent.includes('读取失败'));
      assert.equal($('decision-list').textContent.includes('等待请求'),false);
      fetch=async()=>response({items:[],has_more:false,log_complete:true});await loadDecisions(true);
      assert.ok($('decision-list').textContent.includes('本座位没有可查询'));
      assert.equal($('history-message').hidden,true);
      ui.run={...baseRun,status:'running'};ui.view=state(1);
      fetch=async(path)=>response(path.includes('/decisions?')?{items:[{request_id:'seat-a:q1',opened_seq:1,round_id:1}],has_more:false}:path.endsWith('/state')?state(2):ui.run);
      await pollState();while(ui.decisionsBusy)await Promise.resolve();
      assert.equal(ui.decisions.length,1);
    })()
    """, id="history-loading-error-empty-and-live-arrival"),
    pytest.param(r"""
    (async () => {
      const report={kind:'review',report_id:'report-a',run_id:'run-a',status:'completed',items:[],errors:[]};
      let finish;const calls=[];
      fetch=(path,options={})=>{calls.push([path,options.method||'GET']);return path.endsWith('/reports/report-a')?new Promise(resolve=>finish=resolve):Promise.resolve(response({items:[report],log_complete:true}))};
      const detail=openReport('report-a');await loadReports();finish(response(report));await detail;
      assert.equal(ui.report.report_id,'report-a');assert.equal($('report-body').textContent.includes('正在读取报告'),false);
      assert.ok(calls.every(([,method])=>method==='GET'));
      await $('refresh-reports').listeners.click();
      assert.ok(calls.some(([path,method])=>path.endsWith('/reports/refresh')&&method==='POST'));
      assert.equal(ui.report.report_id,'report-a');
      // A stale detail still cannot overwrite a newly selected run.
      fetch=()=>new Promise(resolve=>finish=resolve);const stale=openReport('report-a');
      ui.runEpoch++;ui.report=null;finish(response(report));await stale;assert.equal(ui.report,null);
    })()
    """, id="report-list-detail-epochs-and-explicit-refresh"),
    pytest.param(r"""
    (async () => {
      const report={kind:'evaluation',report_id:'evaluation-a',run_ids:['run-a'],status:'completed',unit:'match',
        counts:{started:2,settled:1,win:1,loss:0,draw:0,pending:1,interrupted:0},
        performance:{win_rate:{value:1,numerator:1,denominator:1}},
        model_usage:{online:{calls_started:{value:9},total_tokens:{value:null,known_subtotal:80}}},
        quality:{metrics_complete:false}};
      const calls=[];fetch=async(path,options={})=>{calls.push([path,options.method||'GET']);return response(path.endsWith('/evaluate')?report:{items:[report],log_complete:true})};
      ui.reportGenerationEnabled=false;updateControls();await generateReport();assert.equal(calls.length,0);
      assert.ok($('report-generation-hint').textContent.includes('固定演示归档只读'));
      ui.reportGenerationEnabled=true;updateControls();await generateReport();
      assert.ok(calls.some(([path,method])=>path.endsWith('/reports/evaluate')&&method==='POST'));
      assert.equal(ui.report.report_id,'evaluation-a');assert.equal($('generate-report').disabled,false);
      assert.ok($('report-action-status').textContent.includes('模型调用 0 次'));
      assert.ok($('report-body').textContent.includes('100.0%（1 / 1）'));
      assert.ok($('report-body').textContent.includes('原对局总 Tokens—'));
      assert.ok($('report-body').querySelectorAll('details').length > 0);
      ui.run={...baseRun,status:'running'};updateControls();assert.equal($('generate-report').disabled,true);
      ui.run=baseRun;updateControls();fetch=async()=>response({error:{safe_message:'统计不可用'}},false,503);
      await generateReport();assert.ok($('report-action-status').textContent.includes('统计报告生成失败'));
      assert.equal($('generate-report').disabled,false);
    })()
    """, id="explicit-local-statistics-and-unknown-summary-values"),
    pytest.param(r"""
    (async () => {
      const report={kind:'review',report_id:'report-a',run_id:'run-a',status:'completed',items:[],errors:[]};
      let finishDetail,finishList;
      fetch=(path)=>path.endsWith('/reports/report-a')?new Promise(resolve=>finishDetail=resolve)
        :path.endsWith('/reports')?new Promise(resolve=>finishList=resolve)
        :Promise.resolve(response({error:{safe_message:'统计失败'}},false,503));
      ui.reportGenerationEnabled=true;updateControls();
      const detail=openReport('report-a'),list=loadReports();
      await generateReport();finishDetail(response(report));finishList(response({items:[report],log_complete:true}));
      await Promise.all([detail,list]);
      assert.equal(ui.report.report_id,'report-a');
      assert.equal($('report-body').textContent.includes('正在读取'),false);
      assert.equal($('report-list').textContent.includes('正在读取'),false);
      fetch=async()=>response({error:{code:'REPORT_BUSY',safe_message:'已有操作'}},false,409);
      await loadReports();assert.ok($('report-list').textContent.includes('报告操作进行中'));
      assert.equal($('report-list').textContent.includes('运行尚未关闭'),false);
    })()
    """, id="failed-generation-preserves-pending-readers"),
    pytest.param(r"""
    (async () => {
      const report={kind:'review',report_id:'fresh-report',run_id:'run-a',status:'completed'};
      let finish;const calls=[];
      fetch=(path,options={})=>{calls.push([path,options.method||'GET']);return options.method==='POST'
        ?new Promise(resolve=>finish=resolve):Promise.resolve(response({items:[],log_complete:true}))};
      const refresh=loadReports({refresh:true});
      // Returning to the reports tab must not replace an in-flight refresh with an older GET.
      ui.mode='table';setMode('reports');await Promise.resolve();
      assert.equal(calls.length,1);
      finish(response({items:[report],log_complete:true}));await refresh;
      assert.ok($('report-list').textContent.includes('fresh-report'));
      assert.equal($('report-list').textContent.includes('尚未生成'),false);
      assert.equal($('refresh-reports').disabled,false);
    })()
    """, id="tab-return-cannot-overtake-explicit-report-refresh"),
    pytest.param(r"""
    (async () => {
      const seatB={session_id:'seat-b',seat:1,match_id:'match-a'};
      ui.run={...baseRun,sessions:[baseSession,seatB]};
      const capability={enabled:true,model:'deepseek-flash',calls_per_report:1,max_decisions:3,max_calls:5,remaining_calls:5,active_task:null};
      ui.modelReview=capability;updateControls();
      assert.equal($('review-session-select').value,'seat-a');
      $('review-session-select').listeners.change({target:{value:''}});
      assert.equal($('generate-model-review').disabled,true);
      $('review-session-select').listeners.change({target:{value:'seat-b'}});
      assert.equal(ui.session.session_id,'seat-a');
      assert.equal($('generate-model-review').disabled,false);
      let task={task_id:'task-a',source_run_id:'run-a',status:'running',report_ready:false,report_id:null,calls_started:1,slots_used:1,error:null};
      const report={kind:'review',report_id:'model-report-a',run_id:'run-a',status:'completed',items:[],usage:{total_tokens:84},errors:[]};
      const calls=[];
      fetch=async(path,options={})=>{
        calls.push([path,options.method||'GET',options.body]);
        if(path==='/api/v1/configs')return response({items:[],report_generation_enabled:true,model_review:{...capability,remaining_calls:4,active_task:task.report_ready?null:task}});
        if(path.endsWith('/reports/review'))return response(task,true,202);
        if(path.includes('/review-tasks/'))return response(task);
        if(path.endsWith('/reports/model-report-a'))return response(report);
        return response({items:[report],log_complete:true});
      };
      await generateModelReview();await generateModelReview();
      const posts=calls.filter(([,method])=>method==='POST');assert.equal(posts.length,1);
      assert.equal(JSON.parse(posts[0][2]).session_id,'seat-b');
      await pollReviewTasks();assert.equal(calls.some(([path])=>path.endsWith('/reports/model-report-a')),false);
      task={...task,status:'completed',report_id:report.report_id,report_ready:false};
      await pollReviewTasks();assert.equal(ui.report,null);
      task={...task,report_ready:true};await pollReviewTasks();
      assert.equal(ui.report.report_id,'model-report-a');
      assert.ok($('report-body').textContent.includes('Tokens 84'));
      assert.ok($('model-review-hint').textContent.includes('剩余 4 / 5 次'));
      assert.ok($('model-review-status').textContent.includes('座位 1'));
      assert.equal(calls.filter(([,method])=>method==='POST').length,1);
      ui.modelReview={...capability,remaining_calls:0};updateControls();assert.equal($('generate-model-review').disabled,true);
      assert.ok($('model-review-hint').textContent.includes('额度已用完'));
    })()
    """, id="model-review-explicit-seat-one-call-and-publication-gate"),
    pytest.param(r"""
    (async () => {
      ui.modelReview={enabled:true,model:'deepseek-flash',calls_per_report:1,max_decisions:3,max_calls:2,remaining_calls:1};
      const running={task_id:'task-a',source_run_id:'run-a',status:'running',report_ready:false,calls_started:1,slots_used:1};
      rememberReviewTask(running,'seat-a');updateControls();
      ui.reportDetailEpoch++;ui.report={report_id:'manually-selected'}; // User opened another report.
      let task={...running,status:'completed',report_ready:true,report_id:'new-report'};const paths=[];
      fetch=async(path)=>{paths.push(path);return response(path.includes('/review-tasks/')?task:path.endsWith('/configs')?{model_review:ui.modelReview}:{items:[],log_complete:true})};
      await pollReviewTasks();
      assert.equal(ui.report.report_id,'manually-selected');
      assert.equal(paths.some(path=>path.endsWith('/reports/new-report')),false);
      // A report for a run left in the background must not touch the new run.
      const other={...running,task_id:'task-b'};rememberReviewTask(other,'seat-a');
      ui.run={...baseRun,run_id:'run-b'};ui.runEpoch++;ui.report={report_id:'run-b-report'};
      task={...other,status:'completed',report_ready:true,report_id:'background-report'};
      await pollReviewTasks();assert.equal(ui.report.report_id,'run-b-report');
      assert.equal($('model-review-status').hidden,true);
    })()
    """, id="model-task-never-overwrites-newer-report-or-another-run"),
    pytest.param(r"""
    (async () => {
      const saved=JSON.stringify([{source_run_id:'run-a',task_id:'restored-task',session_id:'seat-a'}]);
      localStorage={getItem:(key)=>key==='guandan.review_tasks'?saved:null,setItem(){},removeItem(){}};
      restoreReviewTasks();
      const record=ui.reviewTasks.get('run-a');
      assert.equal(record.sessionId,'seat-a');
      record.watchRunEpoch=ui.runEpoch;record.detailEpoch=ui.reportDetailEpoch;
      ui.reviewSessionId='seat-a';ui.modelReview={enabled:true,remaining_calls:1};
      const task={task_id:'restored-task',source_run_id:'run-a',status:'failed',report_ready:false,calls_started:1,slots_used:1,error:{safe_message:'模型超时'}};
      const calls=[];
      fetch=async(path,options={})=>{calls.push([path,options.method||'GET']);return response(path.includes('/review-tasks/')?task:{model_review:{enabled:true,remaining_calls:0,active_task:null}})};
      await pollReviewTasks();await pollReviewTasks();
      assert.ok(calls.every(([,method])=>method==='GET'));
      assert.equal(calls.filter(([path])=>path.includes('/review-tasks/')).length,1);
      assert.ok($('model-review-status').textContent.includes('模型超时'));
      assert.equal($('generate-model-review').disabled,true);
    })()
    """, id="page-restores-task-with-reads-and-failure-does-not-retry"),
    pytest.param(r"""
    (async () => {
      let finishConfigs;
      const oldCapability={enabled:true,remaining_calls:5};
      fetch=()=>new Promise(resolve=>finishConfigs=resolve);
      const pending=loadConfigs();
      fetch=async()=>response({model_review:{enabled:true,remaining_calls:1},report_generation_enabled:true});
      await loadReviewCapabilities();
      finishConfigs(response({items:[{config_name:'local',policy:'V0',model_calls:0}],model_review:oldCapability,report_generation_enabled:true}));
      await pending;
      assert.equal(ui.configs.length,1);assert.equal(ui.modelReview.remaining_calls,1);
      assert.ok($('config-select').textContent.includes('local'));
      ui.run={...baseRun,read_only:false};ui.runs=[ui.run];updateControls();assert.equal($('start-run').disabled,false);
    })()
    """, id="capability-refresh-cannot-discard-startup-configs-or-restore-old-quota"),
    pytest.param(r"""
    (async () => {
      const task={task_id:'ready-task',source_run_id:'run-a',status:'completed',report_ready:true,report_id:'ready-report'};
      const record=rememberReviewTask(task,'seat-a');ui.reviewSessionId='seat-a';
      let finish;
      fetch=(path)=>path.endsWith('/reports/ready-report')?new Promise(resolve=>finish=resolve):Promise.resolve(response({items:[],log_complete:true}));
      const pending=presentReviewResult(record);
      while(!finish)await Promise.resolve();
      ui.reviewSessionId='seat-b';
      finish(response({kind:'review',report_id:'ready-report',run_id:'run-a',status:'completed',items:[]}));
      await pending;assert.equal(ui.report,null);
      assert.ok($('report-body').textContent.includes('复盘座位已切换'));
      assert.equal($('report-body').textContent.includes('正在读取'),false);
    })()
    """, id="seat-change-during-generated-report-load-cannot-show-old-seat"),
    pytest.param(r"""
    (() => {
      assert.equal(name('invalidated'),'请求已失效');assert.equal(name('deciding'),'正在决策');
      assert.equal(actionName(['Single','B',['SB']]),'单张 · 小王');
      assert.equal(actionName(['Single','R',['HR']]),'单张 · 大王');
      const detail={state:{...state(1).state,binding:{seat:0}},calls:[],
        candidates:{candidates:[{action_id:'L1',features:{played_count:1},keep_reasons:['rule_choice','top_score','legal_pass','opponent_close']}]},
        candidate_actions:[{action_id:'L1',wire_action:['Single','B',['SB']]}],log_complete:true};
      const display=decisionView(detail,{sessionId:'seat-a',requestId:'seat-a:q1'});
      assert.ok(display.textContent.includes('规则首选、规则高分、合法不出、应对对手即将出完'));
      ui.runs=[{...baseRun,label:'B10',run_id:'11111111-rest'},{...baseRun,label:'B10',run_id:'22222222-rest'}];renderRunOptions();
      assert.ok($('run-select').textContent.includes('B10 · 11111111'));assert.ok($('run-select').textContent.includes('B10 · 22222222'));
    })()
    """, id="business-labels-and-distinct-run-options"),
])
def test_ui_read_boundaries(scenario: str) -> None:
    result = subprocess.run(
        [NODE, "-e", HARNESS, str(SCRIPT), scenario],
        text=True, capture_output=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(NODE is None, reason="Node is needed for browser-independent UI checks")
def test_program_explanation_and_unverified_model_reason_are_both_visible():
    scenario = r"""
    const detail={state:{...state(1).state,binding:{seat:0}},calls:[],
      program_explanation:['程序计算：配牌0张'],proposal:{source:'main',action_id:'L0000',reason:'错误原文仍保留'},log_complete:true};
    const display=decisionView(detail,{sessionId:'seat-a',requestId:'seat-a:q1'});
    assert.ok(display.textContent.includes('程序事实说明 · 不评价策略优劣'));
    assert.ok(display.textContent.includes('程序计算：配牌0张'));
    assert.ok(display.textContent.includes('模型原始理由 · 未核验'));
    assert.ok(display.textContent.includes('错误原文仍保留'));
    assert.ok(display.textContent.includes('不代表模型理由已通过核验'));
    """
    result = subprocess.run([NODE, "-e", HARNESS, str(SCRIPT), scenario],
                            text=True, capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
