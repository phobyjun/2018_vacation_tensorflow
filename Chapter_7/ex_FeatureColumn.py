import pandas as pd
import numpy as np

N = 10000

weight = np.random.randn(N) * 5 + 70
spec_id = np.random.randint(0, 3, N)
bias = [0.9, 1, 1.1]
height = np.array([weight[i]/100 + bias[b] for i, b in enumerate(spec_id)])
spec_name = ['Goblin', 'Human', 'ManBears']
spec = [spec_name[s] for s in spec_id]

df = pd.DataFrame({'Species': spec, 'Weight': weight})