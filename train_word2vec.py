import multiprocessing
import os
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

os.environ['CUDA_VISIBLE_DEVICES']='0'
tf_config=tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.per_process_gpu_memory_fraction=0.2
tf_config.gpu_options.allow_growth=True

inp = '/home/lisi/wjh/data/wiki.zh.text.jian.seg.txt'
outp1 = '/home/lisi/wjh/data/wiki.zh.text.model'
outp2 = '/home/lisi/wjh/data/wiki.zh.text.vector'

# inp = 'E:\\python_project\\Chinese_wue\\data\\wiki.zh.text.jian.seg.txt'
# outp1 = 'E:\\python_project\\Chinese_wue\\data\\wiki.zh.text.model'
# outp2 = 'E:\\python_project\\Chinese_wue\\data\\wiki.zh.text.vector'

model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)
