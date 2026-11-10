#!/usr/bin/env python3
"""Project 12 -- Advanced multi-robot coordination demo (SYNTHETIC SOFTWARE).

Compared with the original one-object grid demo, this file introduces:
  * THREE goal objects and a dependency graph: FIND -> REACH -> PICK -> DELIVER;
  * heterogeneous skill restrictions (Go2-like scout vs R1-like executor);
  * an explicit asynchronous observation message bus with per-message latency;
  * timestamp-aware shared memory, invalidation, and live replanning;
  * concurrent exploration and manipulation, plus cost-aware job scheduling;
  * matched-seed comparisons of serial, concurrent FIFO and concurrent cost.

The cost-based scheduler is a simple heuristic; it can make the same choice as
FIFO if only one target has been observed. No learned planner is claimed.

Important limitations: grid-world sensing is deterministic within a radius, robot
navigation is shortest-path grid movement, and picking/delivering are abstract
state transitions. NO VLA, ROS, Unitree dynamics, grasp physics or real data.

Examples:
    python Project12_Advanced_Coordination_Demo.py --episodes 100
    python Project12_Advanced_Coordination_Demo.py --seed 24007 --visualize
    python Project12_Advanced_Coordination_Demo.py --seed 24007 --gif

Requires only Python stdlib for simulation; matplotlib + pillow for plots/GIF.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import random
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Optional

SIDE = 9
BASE = (0, 0)
# Static test geometry, not a calibrated map of a real lab.
WALLS = frozenset({(4, 0), (4, 1), (4, 2), (4, 4), (4, 5), (2, 5), (6, 3)})
OBJECTS = ('bottle', 'cup', 'book')
POLICIES = ('serial', 'concurrent_fifo', 'concurrent_cost')
SENSE_RADIUS = 1
MAX_STEPS = 500


def neighbors(p):
    x, y = p
    for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
        if 0 <= q[0] < SIDE and 0 <= q[1] < SIDE and q not in WALLS:
            yield q


@lru_cache(None)
def path(a, b):
    """Shortest legal grid route, including start and end. Empty if disconnected."""
    if a == b:
        return (a,)
    queue = deque([a]); prev = {a: None}
    while queue:
        u = queue.popleft()
        for v in neighbors(u):
            if v in prev:
                continue
            prev[v] = u
            if v == b:
                result = [b]; p = b
                while prev[p] is not None:
                    p = prev[p]; result.append(p)
                return tuple(reversed(result))
            queue.append(v)
    return ()


def distance(a,b):
    p = path(a,b)
    return len(p)-1 if p else 99999


CELLS = [(x,y) for y in range(SIDE) for x in range(SIDE) if (x,y) not in WALLS]
SWEEP = [p for y in range(SIDE)
         for x in (range(SIDE) if y%2==0 else range(SIDE-1,-1,-1))
         if (p:=(x,y)) not in WALLS]


@dataclass(frozen=True)
class Scenario:
    seed: int
    initial: dict
    final: dict
    relocated: dict
    latency: dict


def scenario(seed):
    rng = random.Random(seed)
    available = [p for p in CELLS if p != BASE and distance(BASE,p) < 99999]
    chosen = rng.sample(available, len(OBJECTS))
    initial = dict(zip(OBJECTS, chosen))
    final = initial.copy(); relocated = {}
    for name in OBJECTS:
        relocated[name] = rng.random() < 0.35
        if relocated[name]:
            final[name] = rng.choice([p for p in available if p not in chosen and p not in final.values()])
    latency = {name: rng.randint(0, 4) for name in OBJECTS}
    return Scenario(seed,initial,final,relocated,latency)


@dataclass(frozen=True)
class Report:
    object_id: str
    position: tuple
    observed_t: int
    arrival_t: int
    sender: str = 'Go2'


@dataclass
class Record:
    position: tuple
    observed_t: int
    valid: bool = True
    source: str = 'Go2'


class Blackboard:
    def __init__(self):
        self.map: dict[str,Record] = {}
        self.accepted = 0
        self.rejected_stale = 0
        self.invalidations = 0

    def receive(self, report):
        old = self.map.get(report.object_id)
        if old is not None and report.observed_t <= old.observed_t:
            self.rejected_stale += 1
            return False
        self.map[report.object_id] = Record(report.position,report.observed_t,True,report.sender)
        self.accepted += 1
        return True

    def invalidate(self, name):
        if name in self.map and self.map[name].valid:
            self.map[name].valid = False
            self.invalidations += 1


@dataclass
class Robot:
    name: str
    skills: frozenset
    pos: tuple = BASE
    steps: int = 0
    trajectory: list = field(default_factory=lambda:[BASE])

    def move_one(self, target):
        route = path(self.pos,target)
        if len(route)>1:
            self.pos=route[1]; self.steps += 1
            self.trajectory.append(self.pos)
            return True
        return False


class Coordinator:
    def __init__(self, mode, scen):
        self.mode = mode; self.scen = scen
        self.now = 0
        self.go2 = Robot('Go2',frozenset({'explore','observe','publish'}))
        self.r1 = Robot('R1',frozenset({'navigate','pick','deliver'}))
        self.memory = Blackboard()
        self.mailbox=[]
        self.mail_serial=0
        self.log=[]
        self.frames=[]
        self.states={name: {'find':'pending','reach':'pending','pick':'pending','deliver':'pending'} for name in OBJECTS}
        self.initial_seen=set()
        self.last_published={}
        self.world_pos=scen.initial.copy()
        self.go2_route_index=0
        self.go2_active_object=OBJECTS[0] if mode=='serial' else None
        self.scout_pause=False
        self.current_job=None
        self.r1_phase='idle'
        self.goal=None
        self.carrying=None
        self.completed=[]
        self.failures=0
        self.replans=0
        self.published=0
        self.motion_time=0
        self.delivery_times=[]
        self.world_moves=0
        self._record('Initialise','Tasks: locate, reach, pick and deliver THREE objects')

    def _record(self,event, note):
        self.log.append({'time':self.now,'event':event,'detail':note,
                         'go2':self.go2.pos,'r1':self.r1.pos,'memory':{k:(v.position,v.valid) for k,v in self.memory.map.items()}})

    def snapshot(self):
        self.frames.append({'time':self.now,'go2':self.go2.pos,'r1':self.r1.pos,
                            'world':self.world_pos.copy(),
                            'memory':{k:(v.position,v.valid) for k,v in self.memory.map.items()},
                            'current_job':self.current_job, 'phase':self.r1_phase,
                            'delivered':tuple(self.completed), 'events':self.log[-1]['detail'] if self.log else ''})

    def send(self,name,pos):
        # Send the source observation, not the true final position.
        latency=self.scen.latency[name]
        packet=Report(name,pos,self.now,self.now+latency)
        self.mail_serial += 1
        heapq.heappush(self.mailbox,(packet.arrival_t,self.mail_serial,packet))
        self.published += 1
        self.last_published[name]=pos
        self._record('Publish',f'Go2 sends {name}@{pos}; latency={latency} steps')
        if name not in self.initial_seen:
            self.initial_seen.add(name)
            if self.scen.relocated[name]:
                self.world_pos[name]=self.scen.final[name]
                self.world_moves += 1
                self._record('Environment change',f'{name} relocated AFTER first observation; no robot notification')

    def deliver_messages(self):
        while self.mailbox and self.mailbox[0][0]<=self.now:
            _,_,msg=heapq.heappop(self.mailbox)
            if self.memory.receive(msg):
                self.states[msg.object_id]['find']='success'
                self._record('Memory update',f'Shared memory accepted {msg.object_id}@{msg.position} (observed t={msg.observed_t})')
            else:
                self._record('Stale rejection',f'Shared memory ignored older {msg.object_id} packet')

    def visible_to_go2(self,name):
        return distance(self.go2.pos,self.world_pos[name])<=SENSE_RADIUS

    def scout_allowed(self):
        if len(self.completed)==len(OBJECTS): return False
        if self.mode!='serial': return True
        if self.go2_active_object in self.completed:
            remaining=[n for n in OBJECTS if n not in self.completed]
            self.go2_active_object=remaining[0] if remaining else None
            self.scout_pause=False
        return not self.scout_pause

    def scout_tick(self):
        if not self.scout_allowed(): return
        if self.mode=='serial':
            targets=[self.go2_active_object] if self.go2_active_object else []
        else:
            targets=[x for x in OBJECTS if x not in self.completed]
        # Observation only within local sensing radius. The planner has NO direct target coordinates.
        for name in targets:
            if self.visible_to_go2(name) and self.last_published.get(name)!=self.world_pos[name]:
                self.send(name,self.world_pos[name])
                if self.mode=='serial':
                    self.scout_pause=True
                    break
        if not self.scout_allowed(): return
        # Sweep all reachable grid cells via legal shortest routes between sweep waypoints.
        for _ in range(len(SWEEP)):
            dest=SWEEP[self.go2_route_index%len(SWEEP)]
            if self.go2.pos != dest:
                self.go2.move_one(dest)
                break
            self.go2_route_index += 1
        # Sensor fires again at next tick; time advances synchronously for both robots.

    def ready_jobs(self):
        return [n for n in OBJECTS if n not in self.completed and n!=self.carrying
                and n!=self.current_job and n in self.memory.map and self.memory.map[n].valid]

    def choose_job(self):
        jobs=self.ready_jobs()
        if not jobs: return None
        if self.mode!='concurrent_cost':
            return min(jobs,key=lambda n:(self.memory.map[n].observed_t,OBJECTS.index(n)))
        # Concrete planning objective: R1's estimated route to the object + delivery base.
        return min(jobs,key=lambda n:(distance(self.r1.pos,self.memory.map[n].position)+
                                       distance(self.memory.map[n].position,BASE), n))

    def start_job(self,name):
        assert 'navigate' in self.r1.skills
        self.current_job=name
        self.r1_phase='approach'
        self.states[name]['reach']='active'
        self.goal=self.memory.map[name].position
        self._record('Assign',f'Coordinator assigns R1: reach {name}@{self.goal}')

    def fail_job(self,name):
        self.failures += 1
        self.states[name]['reach']='failed'
        self.memory.invalidate(name)
        self._record('Failure',f'R1 finds no {name} at {self.r1.pos}; invalidate memory; request Go2 re-observation')
        self.current_job=None;self.goal=None;self.r1_phase='idle'
        self.replans += 1
        self.last_published.pop(name,None)
        if self.mode=='serial':self.scout_pause=False

    def executor_tick(self):
        if self.r1_phase=='idle':
            job=self.choose_job()
            if job is not None:self.start_job(job)
        if self.r1_phase=='approach':
            name=self.current_job
            memory=self.memory.map.get(name)
            if memory is not None and memory.valid and memory.position != self.goal:
                self.goal=memory.position
                self.replans+=1
                self._record('Replan',f'R1 updates {name} destination from new shared-memory observation')
            if self.goal is not None and self.r1.pos!=self.goal:
                self.r1.move_one(self.goal)
            if self.goal is not None and self.r1.pos==self.goal:
                # This is an ABSTRACT verification/pick outcome, not Unitree manipulation.
                if self.world_pos[name] != self.r1.pos:
                    self.fail_job(name)
                else:
                    assert 'pick' in self.r1.skills
                    self.states[name]['reach']='success'
                    self.states[name]['pick']='success'
                    self.carrying=name
                    self.r1_phase='deliver'
                    self._record('Pick',f'R1 symbolically picks up {name} at {self.r1.pos}')
        elif self.r1_phase=='deliver':
            name=self.carrying
            if self.r1.pos!=BASE:
                self.r1.move_one(BASE)
            if self.r1.pos==BASE:
                assert 'deliver' in self.r1.skills
                self.states[name]['deliver']='success'
                self.completed.append(name)
                self.delivery_times.append(self.now)
                self._record('Deliver',f'R1 delivered {name}; {len(self.completed)}/{len(OBJECTS)} completed')
                self.carrying=None;self.current_job=None;self.goal=None
                self.r1_phase='idle'

    def run(self,keep_frames=False):
        for t in range(MAX_STEPS):
            self.now=t
            self.deliver_messages()
            # Simultaneous actions in the same step, one move per robot.
            self.executor_tick()
            self.scout_tick()
            # Important for latency=0: deliver newly created report after scout tick too.
            self.deliver_messages()
            if keep_frames:self.snapshot()
            if len(self.completed)==len(OBJECTS):break
        success=int(len(self.completed)==len(OBJECTS))
        makespan=self.now+1
        return {'seed':self.scen.seed,'policy':self.mode,'success':success,
                'delivered':len(self.completed),'makespan':makespan,
                'r1_steps':self.r1.steps,'go2_steps':self.go2.steps,
                'team_steps':self.r1.steps+self.go2.steps,
                'messages':self.published,'failures':self.failures,
                'replans':self.replans,'relocations':sum(self.scen.relocated.values()),
                'stale_rejections':self.memory.rejected_stale}


def compare(episodes, start, output):
    rows=[]
    for i in range(episodes):
        sc=scenario(start+i)
        for policy in POLICIES:
            c=Coordinator(policy,sc)
            rows.append(c.run())
    summary=[]
    for policy in POLICIES:
        subs=[r for r in rows if r['policy']==policy]
        summary.append({'policy':policy,'episodes':len(subs),
                        'task_success_pct':100*mean(x['success'] for x in subs),
                        'mean_makespan':mean(x['makespan'] for x in subs),
                        'mean_team_steps':mean(x['team_steps'] for x in subs),
                        'mean_r1_steps':mean(x['r1_steps'] for x in subs),
                        'mean_go2_steps':mean(x['go2_steps'] for x in subs),
                        'mean_messages':mean(x['messages'] for x in subs),
                        'mean_failures':mean(x['failures'] for x in subs),
                        'mean_replans':mean(x['replans'] for x in subs)})
    output.mkdir(parents=True,exist_ok=True)
    for fname,data in [('advanced_episode_results.csv',rows),('advanced_summary.csv',summary)]:
        with (output/fname).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    return rows,summary


def plot_summary(summary,output):
    import matplotlib.pyplot as plt
    names=['Serial handoff','Concurrent FIFO','Concurrent cost-aware']
    fig, ax=plt.subplots(figsize=(9,4))
    x=list(range(len(names)))
    vals=[v['mean_makespan'] for v in summary]
    bars=ax.bar(x,vals,color=['white','#cccccc','#888888'],edgecolor='black')
    for bar,hatch in zip(bars,('', '///', 'xx')): bar.set_hatch(hatch)
    for n,v in zip(x,vals):ax.text(n,v+1,f'{v:.1f}',ha='center',fontsize=10)
    ax.set_xticks(x,names);ax.set_ylabel('Mean completion time (abstract steps)')
    ax.set_title('Heterogeneous task coordination (synthetic grid experiment)')
    ax.grid(axis='y',alpha=.3);fig.tight_layout()
    path_out=output/'advanced_makespan.png'; fig.savefig(path_out,dpi=170);plt.close(fig)
    return path_out


def visualize(c, save_path, show, fps=8):
    """Animation of the REAL event trace emitted by this synthetic simulator."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Rectangle
    except ImportError as e:
        raise RuntimeError('Visualisation requires: pip install matplotlib pillow') from e
    if not c.frames:
        raise ValueError('Run Coordinator.run(keep_frames=True) first')

    fig,(ax,info)=plt.subplots(1,2,figsize=(11.5,6.3),gridspec_kw={'width_ratios':[1.15,1]})
    ax.set_xlim(-.5,SIDE-.5); ax.set_ylim(-.5,SIDE-.5)
    ax.set_aspect('equal'); ax.set_xticks(range(SIDE)); ax.set_yticks(range(SIDE))
    ax.grid(alpha=.17)
    for x,y in WALLS:
        ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor='#666',edgecolor='#222'))
    ax.plot(BASE[0],BASE[1],'D',color='black',markersize=9,label='Delivery base')
    go,=ax.plot([],[],'o',color='#1f77b4',markersize=12,label='Go2-like scout')
    r1,=ax.plot([],[],'s',color='#ed7d31',markersize=11,label='R1-like executor')
    target,=ax.plot([],[],'*',color='#bc4141',markersize=14,label='True objects (viewer only)')
    mem,=ax.plot([],[],'x',color='#222',markersize=11,mew=2,label='Shared-memory targets')
    ax.legend(loc='upper right',fontsize=7,framealpha=.93)
    ax.set_title('Synthetic Go2 / R1 grid-world',fontsize=12)
    info.axis('off')
    status=info.text(0,1,'',va='top',ha='left',fontsize=11,fontfamily='monospace',linespacing=1.55)
    fig.text(.05,.035,'Ground truth is shown for the viewer; the coordinator receives only observation messages.',fontsize=9)
    texts={name:ax.text(0,0,'',fontsize=9,color='#7f2525',fontweight='bold') for name in OBJECTS}

    def render(i):
        s=c.frames[i]
        go.set_data([s['go2'][0]],[s['go2'][1]])
        r1.set_data([s['r1'][0]],[s['r1'][1]])
        remaining=[(name,p) for name,p in s['world'].items() if name not in s['delivered']]
        target.set_data([p[0] for _,p in remaining],[p[1] for _,p in remaining])
        known=[p for name,(p,valid) in s['memory'].items() if valid and name not in s['delivered']]
        mem.set_data([p[0] for p in known],[p[1] for p in known])
        for name,label in texts.items():
            q=s['world'][name]
            label.set_position((q[0]+.2,q[1]+.12))
            label.set_text('' if name in s['delivered'] else name)
        objects='\n'.join(f" {name:<7} {'DELIVERED' if name in s['delivered'] else ('known' if name in s['memory'] and s['memory'][name][1] else 'unknown')}" for name in OBJECTS)
        status.set_text(f"t = {s['time']}  |  {c.mode}\n"
                        f"----------------------------------\n"
                        f"Go2  {str(s['go2']):<8}  [explore / observe]\n"
                        f"R1   {str(s['r1']):<8}  [{s['phase']}]\n\n"
                        f"TASK GRAPH per object:\n"
                        f"  FIND -> REACH -> PICK -> DELIVER\n\n"
                        f"SHARED MEMORY STATUS:\n{objects}\n\n"
                        f"Current R1 job: {s['current_job'] or '-'}\n"
                        f"Delivered: {len(s['delivered'])}/{len(OBJECTS)}\n\n"
                        f"Latest event:\n{s['events'][:48]}\n{s['events'][48:96]}")
        return (go,r1,target,mem,status,*texts.values())

    stride=max(1,len(c.frames)//180)
    ids=list(range(0,len(c.frames),stride))
    if ids[-1]!=len(c.frames)-1:
        ids.append(len(c.frames)-1)
    anim=FuncAnimation(fig,render,frames=ids,interval=1000/fps,blit=False,repeat=False)
    fig.subplots_adjust(left=.045,right=.985,bottom=.14,top=.93,wspace=.15)
    if save_path:
        out=Path(save_path);out.parent.mkdir(parents=True,exist_ok=True)
        anim.save(out,writer=PillowWriter(fps=fps))
        print(f'Animation saved: {out} ({len(ids)} frames)')
    if show:
        plt.show()
    else:
        plt.close(fig)


def check():
    assert all(distance(BASE,p)<99999 for p in CELLS)
    for seed in (100,101,102,103,104):
        sc=scenario(seed)
        assert all(sc.initial[k]!=sc.final[k] for k in OBJECTS if sc.relocated[k])
        for p in POLICIES:
            run=Coordinator(p,sc)
            report=run.run()
            assert report['delivered']<=len(OBJECTS)
            assert report['messages']>=0
            assert len(run.completed)==len(set(run.completed))
            if report['success']:
                assert all(run.states[k]['deliver']=='success' for k in OBJECTS)
            assert run.r1.skills != run.go2.skills
    # Verify timestamp ordering is by observation time, not delivery time.
    mem=Blackboard()
    assert mem.receive(Report('bottle',(5,3),10,12))
    assert not mem.receive(Report('bottle',(1,1),7,15))
    assert mem.map['bottle'].position==(5,3)
    print('Basic correctness checks: PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=24003)
    p.add_argument('--out',type=Path,default=Path('advanced_results'))
    p.add_argument('--visualize',action='store_true',help='show one simulated episode in an interactive window')
    p.add_argument('--gif',action='store_true',help='save a GIF of one simulated episode')
    p.add_argument('--policy',choices=POLICIES,default='concurrent_cost')
    args=p.parse_args()
    if args.episodes<1:p.error('--episodes must be positive')
    check()
    _,s=compare(args.episodes,args.seed,args.out)
    print('Policy               Success%    Makespan   Team steps   R1 steps   Messages')
    for x in s:
        print(f"{x['policy']:<21} {x['task_success_pct']:7.1f} {x['mean_makespan']:11.1f} "
              f"{x['mean_team_steps']:12.1f} {x['mean_r1_steps']:10.1f} {x['mean_messages']:10.2f}")
    try:print('Summary plot:',plot_summary(s,args.out))
    except ImportError:print('matplotlib not installed; CSV results saved (pip install matplotlib pillow)')
    if args.visualize or args.gif:
        c=Coordinator(args.policy,scenario(args.seed))
        result=c.run(keep_frames=True)
        print('Illustrative episode:',result)
        for e in c.log:
            if e['event'] in ('Publish','Memory update','Failure','Deliver','Replan','Environment change'):
                print(f"  t={e['time']:3d} {e['event']:<19} {e['detail']}")
        visualize(c,args.out/(f'advanced_{args.policy}_seed{args.seed}.gif') if args.gif else None,args.visualize)


if __name__=='__main__':main()
