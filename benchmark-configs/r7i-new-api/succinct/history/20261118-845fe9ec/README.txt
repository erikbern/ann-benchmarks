r7i：succinct 三数据集最佳已测配置（2026-11-18）

适用分支：feat/quantized-graph-performance-exploration-succinct；复验代码：98a265b0。
归档目录按 r7i 命名；本次虚拟机读取的 CPU 为 Intel Xeon Platinum 8369B。
SIFT/GIST/GloVe 均为 FP32、TopK=10、单核查询；配置覆盖 ANN Recall@10 的 30/50/70/85/90/95/99% 下限。

文件与数值口径
sift.json、gist.json、glove.json：各档构建配置、查询参数、YAML case 索引、原始结果及索引指纹。
best_observed 为所有纳入记录中达到召回下限的最高正式 QPS；mean_query_time_us=1e6/QPS。
它是不同批次的历史最好值，不能作为同一批次排名或性能保证。
revalidation 给出该配置本轮两次 best-of-five 的结果；same_session_best 单独给出本轮中位数最快的配置。
主 query_config/query_case_index 指向 best_observed；same_session_best 有自己的配置引用。case 索引从0开始。
summary.csv 汇总三数据集；validation.json 记录 succinct 二进制、复验协议与返回 ID 检查。
结果路径以外部 ann-benchmarks 工程为根；本目录不打包大数据集、索引或 HDF5 结果文件。

使用
1. 替换 YAML 中 /data/<dataset>/ 路径。BuildFile 必须是 local_builder 的 zvec.vecs 格式，不能直接使用 HDF5/fvecs。
2. 按 JSON profiles.<profile>.build_configs 的顺序调用已编译的 local_builder：
   ./build-ann/bin/local_builder tools/core/bench/config/r7i/sift_base_build.yaml
   ./build-ann/bin/local_builder tools/core/bench/config/r7i/sift_n64_build.yaml
   GIST 只需 gist_n64_build.yaml；GloVe 先 glove_base_build.yaml，再派生 glove_n32_build.yaml / glove_n64_build.yaml。
3. 搜索构建配置的 DumpPath；查询 YAML 的 IndexPath 与之对应。
   QueryParams 优先于 QueryParam；要只测一档，可只保留所选 case，或移除 QueryParams 并设置 QueryParam。
4. native bench/recall 工具可加载查询 YAML；它们的 QPS 计时口径与 ANN-Benchmarks 不同。
   native recall 的 CompareById=true 报告严格 ID 召回；本文目标采用第10真值距离+0.001，二者不能混用。

构建与迁移边界
GloVe 32/64 邻居是不同的量化图构建配置，不能仅修改查询参数切换。两种布局从同一原始 HNSW 图派生。
本轮在 succinct 二进制上加载已有不可变索引并复验全部候选；部分索引最初由 exploration 构建。
新导入的 GloVe 参数先在 succinct 上完成正式复测，JSON 中不把 exploration 的 QPS 当成 succinct 结果。
SIFT 使用两旋转 native 编码；GIST 使用单旋转 low_rank 编码。请保留完整构建参数及路由参数。
所有候选返回 ID 与原始记录一致；重建图、换机器或更改数据后仍需重新校准召回。
公开查询集同时用于调参和报告；本目录代表已测范围的最佳配置，不声明全局最优。

数据集 / 召回下限 / 最快实测QPS / 平均查询微秒 / 索引布局
sift-128-euclidean / 30% / 73703 / 13.568 / n64
sift-128-euclidean / 50% / 61135 / 16.357 / n64
sift-128-euclidean / 70% / 49273 / 20.295 / n64
sift-128-euclidean / 85% / 43727 / 22.869 / n64
sift-128-euclidean / 90% / 43727 / 22.869 / n64
sift-128-euclidean / 95% / 36073 / 27.722 / n64
sift-128-euclidean / 99% / 23842 / 41.943 / n64
gist-960-euclidean / 30% / 33510 / 29.842 / n64
gist-960-euclidean / 50% / 27053 / 36.965 / n64
gist-960-euclidean / 70% / 20850 / 47.962 / n64
gist-960-euclidean / 85% / 14101 / 70.915 / n64
gist-960-euclidean / 90% / 13544 / 73.835 / n64
gist-960-euclidean / 95% / 9198 / 108.723 / n64
gist-960-euclidean / 99% / 3360 / 297.579 / n64
glove-100-angular / 30% / 60790 / 16.450 / n32
glove-100-angular / 50% / 48042 / 20.815 / n32
glove-100-angular / 70% / 31696 / 31.549 / n32
glove-100-angular / 85% / 18586 / 53.805 / n64
glove-100-angular / 90% / 15313 / 65.303 / n64
glove-100-angular / 95% / 9329 / 107.194 / n64
glove-100-angular / 99% / 3815 / 262.154 / n64
