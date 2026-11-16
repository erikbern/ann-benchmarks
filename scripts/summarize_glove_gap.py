"""Summarize verified local effects without inferring unknown c9i provenance."""
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from diagnose_glove_gap import OUT, PRIOR, ROOT, save


def main():
    reference=json.loads((OUT/'c9i-reference.json').read_text())
    protocols=json.loads((OUT/'protocols.json').read_text())
    confirmations=json.loads((OUT/'confirmations.json').read_text())
    graphs=json.loads((OUT/'graphs.json').read_text())
    method=json.loads((OUT/'method.json').read_text())
    protocol_summary=[]
    tuned=[]
    for target in [.9,.95,.99]:
        row={'target':target}
        for mode in ['native','python','interleaved','official']:
            matches=[r for r in protocols if r['target']==target and r['mode']==mode]
            assert len(matches)==2
            row[mode+'_qps']=median(r['qps_best'] for r in matches)
            row[mode+'_repeat_qps']=[r['qps_best'] for r in matches]
        row['python_vs_native_pct']=100*(row['python_qps']/row['native_qps']-1)
        row['interleaved_vs_python_pct']=100*(row['interleaved_qps']/row['python_qps']-1)
        protocol_summary.append(row)
        groups=defaultdict(list)
        for r in confirmations:
            if r['target']==target: groups[json.dumps(r['parameters'],sort_keys=True)].append(r)
        candidates=[]
        for params,records in groups.items():
            assert len(records)==2 and len({r['ids_sha256'] for r in records})==1
            candidates.append(dict(parameters=json.loads(params),qps=median(r['qps'] for r in records),
                repeat_qps=[r['qps'] for r in records],recall=records[0]['recall'],
                strict_id_recall=records[0]['strict_id_recall']))
        original=next(p['parameters'] for p in method['selected_parameters'] if p['target']==target)
        base=next(r for r in candidates if r['parameters']==original)
        best=max((r for r in candidates if r['recall']>=target),key=lambda r:r['qps'])
        remote=reference['glove']['succinct'][[.9,.95,.99].index(target)]
        tuned.append(dict(target=target,baseline=base,best=best,gain_pct=100*(best['qps']/base['qps']-1),
                          c9i_qps=remote,vs_c9i_pct=100*(best['qps']/remote-1)))
    save('protocol-summary.json',protocol_summary)
    save('tuning-summary.json',tuned)

    graph_summary=[]
    for target in [.9,.95,.99]:
        row={'target':target}
        for index in sorted({r['index'] for r in graphs}):
            matches=[r for r in graphs if r['target']==target and r['index']==index]
            assert len(matches)==2
            label=Path(index).parent.name
            row[label]=dict(native_qps=median(r['native_qps'] for r in matches),
                repeat_qps=[r['native_qps'] for r in matches],recall=matches[0]['recall'],
                strict_id_recall=matches[0]['strict_id_recall'])
        graph_summary.append(row)
    save('graph-summary.json',graph_summary)

    stages=[]
    point=mode=None
    for line in (OUT/'profile.log').read_text().splitlines():
        if line.startswith('PROFILE_POINT '):
            point=json.loads(line[len('PROFILE_POINT '):]);mode='warmup'
        elif line.startswith('PROFILE_MODE '): mode=line.split()[1]
        elif line.startswith('GLOVE_PROFILE '):
            values=list(map(float,line.split()[1:]))
            assert len(values)==8
            stages.append(dict(**point,mode=mode,**dict(zip([
                'prepare_us','router_us','traversal_us','refinement_us','completion_us',
                'expanded','refine_candidates','exact_refinements'],values))))
    assert stages
    save('stages.json',stages)
    stage_groups=defaultdict(list)
    for r in stages:
        if r['mode']!='warmup':
            stage_groups[(r['target'],r['parameters']['quantized_graph_margin'],r['mode'])].append(r)
    stage_summary=[]
    for (target,margin,mode),records in stage_groups.items():
        means={k:sum(r[k] for r in records)/len(records) for k in [
            'prepare_us','router_us','traversal_us','refinement_us','completion_us',
            'expanded','refine_candidates','exact_refinements']}
        stage_summary.append(dict(target=target,margin=margin,mode=mode,**means))
    save('stage-summary.json',stage_summary)
    save('validated-query-presets.json',dict(
        index=method['index'],binary_sha256=method['binary_sha256'],topk=10,query_threads=1,
        scope='Validated on this fixed local graph and the full public query set; recalibrate after rebuilding the graph.',
        presets=[dict(target=r['target'],**r['best']) for r in tuned]))

    remote_all={'sift':[48491,39618,25220],'gist':[15617,10437,3507],'glove':[16345,9633,3182]}
    cross=[]
    for r in json.loads((PRIOR/'compact-official-summary.json').read_text()):
        d=r['dataset'].split('-')[0];remote=remote_all[d][[.9,.95,.99].index(r['target'])]
        local=r['compact_qps']
        cross.append(dict(dataset=d,target=r['target'],c9i_qps=remote,local_qps=local,
            delta_pct=100*(local/remote-1),extra_microseconds=1e6/local-1e6/remote))
    save('cross-dataset-gap.json',cross)
    vendor=[]
    prior_rows=list(csv.DictReader((ROOT/'benchmarks/succinct-diagnosis-20261116/four-algorithm-comparison.csv').open()))
    for r in prior_rows:
        if r['dataset']!='glove-100-angular' or r['algorithm']=='exploration': continue
        name='succinct' if r['algorithm']=='succinct-optimized' else r['algorithm']
        target=float(r['target']);remote=reference['glove'][name][[.9,.95,.99].index(target)]
        vendor.append(dict(algorithm=name,target=target,c9i_qps=remote,local_qps=float(r['qps']),
                          local_recall=float(r['recall']),delta_pct=100*(float(r['qps'])/remote-1)))
    save('relative-ranking.json',vendor)

    lines=[
        'GloVe 跨机器差距：本机可重复验证的因素',
        '',
        '已验证的可修正因素：候选参数网格较粗，margin 偏保守，产生了超过目标召回所需的图遍历。',
        '只调整 margin，90/95/99% 三档正式 QPS 分别提高约5.81%/7.51%/23.03%。',
        '插桩显示图遍历占核心查询时间约91%-96%；99%档平均展开节点从630.779降至500.430，精排向量仍约12个。',
        '当前两份图的吞吐差异小于0.5%；CPU10与CPU48近似，CPU0反而略慢，CPU48选择不能解释大幅低速。',
        '99%档细化结果高于用户c9i表值；90%与95%仍分别低9.47%和4.36%，剩余跨机器差异尚不能定量归因。',
        '参数收益改变了查询配置，不能据此声称已解释同参数跨机器差距，也不代表c9i无法继续调优。',
        '',
        '当前 CPU：Intel Xeon Platinum 8369B；用户称当前为 r9i，但本会话只能确认 CPU，不能确认云实例 SKU。',
        '参照：用户提供的 c9i 汇总表。未取得 c9i CPU 型号、逐点实际召回、计时实现、原始索引与源码/二进制版本。',
        '因此以下是本机受控消融结果；不能把本机某个效应自动归为 c9i/r9i 跨机器差距的全部原因。',
        'Git 历史 3354eb11 的 11月3日归档记录 Xeon 6982P-C，L2每核2MiB、L3报告504MiB；当前每核L2为1.25MiB、L3报告48MiB。',
        '该历史归档并非用户本次汇总表的已确认出处；虚拟机报告的共享L3大小也不等于实例独占缓存。',
        '历史归档 runner.py SHA256 与本机完全相同；旧报告取5轮中位数，本机取5轮最快。不能声称已证实远端使用原生循环。',
        '',
        '方法：GloVe 全部 1,183,514 个 FP32 向量、全部 10,000 个查询，TopK=10。',
        '固定 CPU48，OMP/BLAS=1；暂停其他评测调度，等待已有 worker 完成后串行计时，结束自动恢复。',
        '复用同一个完整索引和已验证 compact 二进制。所有性能参数实验保持算法源码不变。',
        '正式 QPS：官方 run_individual_query 每进程 5 轮取最快，正反顺序 2 次观察取中位数。',
        '召回：ANN-Benchmarks 距离阈值（第10邻居距离+0.001）；严格 ID 召回另列，二者不混用。',
        '参数在同一公开查询集调优及报告，不代表独立测试集效果。',
        '',
        '原参数 -> margin 细化（仅在已验证候选中选择，非全局最优）',
        '目标    原 QPS/实际召回       新 QPS/实际召回       提升       c9i QPS    与c9i差异',
    ]
    for r in tuned:
        a,b=r['baseline'],r['best']
        lines.append(f"{r['target']:.0%}  {a['qps']:9.1f}/{a['recall']:.5f}  {b['qps']:9.1f}/{b['recall']:.5f}  {r['gain_pct']:+.2f}%  {r['c9i_qps']:7d}  {r['vs_c9i_pct']:+.2f}%")
        lines.append('  新参数 '+json.dumps(b['parameters'],ensure_ascii=False,sort_keys=True))
    lines+=['','计时方式对照：同索引、参数、完整查询集合；各方式都取5轮最快后合并2次观察。',
            'native 是原生循环；python 包含 Python 单次调用但不交错距离校验；interleaved 交错距离校验；official 是原版运行器。',
            '距离校验本身不在查询计时区间内。差值反映调用/交错负载的影响，不证明具体微架构机制，更不证明 c9i 使用了哪种计时方式。']
    for r in protocol_summary:
        lines.append(f"{r['target']:.0%}: native={r['native_qps']:.1f}, python={r['python_qps']:.1f}, interleaved={r['interleaved_qps']:.1f}, official={r['official_qps']:.1f}")
    lines+=['','三数据集原参数相对 c9i 的 Zvec 降幅（原表口径，参数/原始索引跨机器未核实）：']
    for r in cross:
        lines.append(f"{r['dataset']} {r['target']:.0%}: {r['delta_pct']:+.2f}%, 查询延迟差 {r['extra_microseconds']:+.3f} us")
    lines+=['','GloVe 三算法相对 c9i：']
    for r in vendor:
        lines.append(f"{r['algorithm']} {r['target']:.0%}: {r['delta_pct']:+.2f}%, 本机实际召回 {r['local_recall']:.5f}")
    lines+=['','阶段计时使用独立插桩构建，只用于热点归因，不用于性能排名。完整返回 ID 与未插桩版本逐项一致。',
            'stage-summary.json 给出查询准备、路由、图遍历、精排的平均微秒和展开/精排次数。',
            '精排阶段包含候选排序与向量读取，并非仅浮点距离计算。',
            'graphs.json 对照两份本机独立构建的索引；两者都不是 c9i 原始索引，不能据此排除 c9i 索引差异。',
            'mappings.json 记录本轮大页映射；既有 GloVe arena 大页覆盖率约99.97%，旧日志 coverage=-1 为无法探测，不是大页缺失。',
            '',
            '仍需 c9i 原始证据：实际 CPU、二进制/源码 SHA、CampaignQG/campaign.py、逐点实际召回和轮次耗时、索引构建配置及快照。',
            '本地 plan_succinct.json 只有参数，不能证明远端表格的获胜配置或计时协议。',
            '特别是 plan 的 build_threads=1，而本机既有 GloVe 构建为8线程；没有远端索引时不能把此项当作已排除。',
            '没有新增生产算法改动；本轮不向 succinct 分支重新添加结果目录。']
    lines+=['','核心阶段计时（native均值，单位微秒；统计不含外层Python/HNSW封装）：',
            '目标 margin 准备 路由 遍历 精排 完成 展开节点 精排向量']
    for r in stage_summary:
        if r['mode']=='native':
            lines.append(f"{r['target']:.0%} {r['margin']} "+' '.join(f'{r[k]:.3f}' for k in [
                'prepare_us','router_us','traversal_us','refinement_us','completion_us','expanded','exact_refinements']))
    lines+=['','核绑定对照（相同原参数，native三轮中位数；CPU48前后夹测CPU10和CPU0）：']
    affinity=json.loads((OUT/'cpu-affinity.json').read_text())
    assert len(affinity)==8
    for r in affinity:
        lines.append(f"{r['target']:.0%} CPU{r['cpu']} position={r['position']}: {r['native_qps']:.1f} QPS")
    lines+=['','两份本机图使用相同已优化二进制及原查询参数（native两次观察中位数）：']
    for r in graph_summary:
        a,b=r['glove-exploration-v2'],r['glove-succinct-v2']
        lines.append(f"{r['target']:.0%}: exploration图 {a['native_qps']:.1f}/{a['recall']:.5f}, succinct图 {b['native_qps']:.1f}/{b['recall']:.5f}, QPS差 {100*(b['native_qps']/a['native_qps']-1):+.3f}%")
    (OUT/'report.zh-CN.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:24]))


if __name__=='__main__':
    main()
