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
      fetch=async(path)=>response(path.endsWith('/state')?state(3):baseRun); await pollState(); assert.equal(ui.view.view_version,3);
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
      for(const value of [
        {...modern,items:[{...item,facts_source:'program',facts_verified:false}]},
        {...modern,generation:{prompt_version:'b09-offline-v1'}},
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
      fetch=async()=>response({error:{code:'HISTORY_UNAVAILABLE',safe_message:'历史已丢失'}},false,410,{'X-Retained-From-Seq':'8'});
      await loadDecisions(true);assert.equal($('recover-history').hidden,false);assert.equal(ui.retainedFrom,8);
      let path;fetch=async(url)=>{path=url;return response({items:[],has_more:false,next_after_seq:null,retained_from_seq:8,log_complete:false})};
      await $('recover-history').listeners.click();assert.ok(path.includes('after_seq=7'));
      assert.equal($('history-message').hidden,false);
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
      assert.ok(display.textContent.includes('Pair · 3'));assert.equal(display.querySelectorAll('span').some(node=>node.attrs['aria-label']==='♥3'),true);
      await selectRun('');assert.equal(ui.run,null);assert.equal($('start-run').disabled,false);
    })()
    """, id="run-summary-refreshes-in-reports-and-keeps-seat"),
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
