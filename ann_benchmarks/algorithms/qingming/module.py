import numpy as np
import ann_benchmarks.algorithms.qingming.qingming_hip as engine
from ann_benchmarks.algorithms.base.module import BaseANN

class QingMing(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = metric
        self.k = method_param.get("k", 100)
        # 必须与 config.yml 的 name 一致
        self.name = "qingming"
        # [Strict Protocol] 初始化结果容器
        self.res = []

    def fit(self, X):
        print(f"[{self.name}] Indexing {X.shape} on AMD GPU...")
        
        if self.metric == 'angular':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
            
        self._data = np.array(X, dtype=np.float32, order='C')
        self.engine = engine.QingMingEngine(self._data)

    def set_query_arguments(self, k):
        self.k = k

    def batch_query(self, X, n):
        # [Strict Protocol] 只计算，不返回，存入 self.res
        if self.metric == 'angular':
             norms = np.linalg.norm(X, axis=1, keepdims=True)
             norms[norms == 0] = 1.0
             X = X / norms
        
        X = np.array(X, dtype=np.float32, order='C')
        
        # C++ 返回的是 numpy array
        # 官方框架有时需要 list of list，但 numpy array 通常也兼容
        # 为了稳妥，如果你之前的 .so 返回的是 numpy array，这里直接赋值
        self.res = self.engine.query(X, n)

    # get_batch_results 使用基类的默认实现即可（它就是 return self.res）
    # 所以不需要重写它

    def query(self, v, n):
        # Latency 模式
        if self.metric == 'angular':
            v = v / np.linalg.norm(v)
        v_batch = np.array([v], dtype=np.float32)
        return self.engine.query(v_batch, n)[0]

    def __str__(self):
        return self.name
