#coding:utf-8
#train,dev,test的最长有51个词
import tensorflow as tf
import re
import numpy as np
import os
import random
import gensim
import codecs

os.environ['CUDA_VISIBLE_DEVICES']='0'
tf_config=tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.per_process_gpu_memory_fraction=0.2
tf_config.gpu_options.allow_growth=True

traindir='/home/lisi/wjh/data/train.txt'
devdir='/home/lisi/wjh/data/dev.txt'
testdir='/home/lisi/wjh/data/test.txt'
Word2Vecdir='/home/lisi/wjh/data/wiki.zh.text.model'
modelpath='/home/lisi/wjh/result/lstmsimple/'

resultfile=codecs.open(modelpath+'result.txt','a',encoding='utf-8')


#读取train文件 返回为 train_text和train_label，两个二维数组，错误处标记1，正确为0
def load_corpus(traindir,text_len):
    train_text = []
    train_label= []
    a = re.compile(u'[<][\u4e00-\u9fa5]+[>]')
    lines = codecs.open(traindir,'r', encoding='UTF-8').readlines()
    for line in lines:
        train_text_line = []
        train_label_line = []
        line = line.replace('\n', '').split(' ')
        for word in line:
            if a.match(word):
                word=re.sub('[<>]','',word)
                train_text_line.append(word)
                train_label_line.append(1)
            else:
                train_text_line.append(word)
                train_label_line.append(0)
            if len(train_text_line)>=text_len:          #句子不等长··
                break
        if len(train_text_line) < text_len:
            for i in range(len(train_text_line), text_len):
                train_text_line.append('<a>')                                            #每行剩下的用<a>来补
                train_label_line.append(0)

        train_text.append(train_text_line)
        train_label.append(train_label_line)
    return train_text,train_label

#字典
def build_vocab(train_text):
    code=int(0)
    vocabdict={}
    vocabdict['UNKNOWN']=code
    code+=1
    for line in train_text:
        for word in line:
            if len(word)<=0:
                continue
            if word not in vocabdict :
                vocabdict[word]=code
                code +=1
    return vocabdict

def encode(vocabdict,line):
    x=[]
    for word in line:
        if word in vocabdict :
            x.append(vocabdict[word])
        else:
            x.append(vocabdict['UNKNOWN'])
    return x

#提取出batch_size句话
def load_train_data(batch_size,train_text,train_label,vocabdict):
    """
    train_x  [[8,9,10,..]       每行一句话，数字为在字典中标号
              [6,7,8,...]
              [4,5,6,...]]
    """
    train_x=[]
    train_y=[]
    for i in range(0,batch_size):
        rng=random.randint(0,len(train_text)-1)
        line=train_text[rng]
        train_x.append(encode(vocabdict,line))
        train_y.append(train_label[rng])
    return np.array(train_x,dtype='float32'),np.array(train_y,dtype='int32')


def load_word_embeddings(vocabdict,embedding_size,train_text):      #-------------vector文件
    embeddings=[]
    for i in range(0,len(vocabdict)):
        vec=[]
        for j in range(0,embedding_size):
            vec.append(0.01)
        embeddings.append(vec)
    model = gensim.models.Word2Vec.load(Word2Vecdir)
    for line in train_text:
        for word in line:
            if word in model:
                id=vocabdict[word]
                embeddings[id]=model[word]

    return tf.Variable(np.array(embeddings,dtype='float32'),dtype=tf.float32)

### test 部分 ###
def load_test_data(batch_size,test_text,test_label,vocabdict,n):
    test_x=[]
    test_y=[]
    total_n_test=len(test_label)
    if n+batch_size<total_n_test:
        for i in range(batch_size):
            line=test_text[i+n]
            test_x.append(encode(vocabdict, line))
            test_y.append(test_label[i+n])
        n=n+batch_size
        return np.array(test_x,dtype='float32'),np.array(test_y,dtype='int32'),n,n
    else:
        for i in range(total_n_test-n):
            line = test_text[i + n]
            test_x.append(encode(vocabdict, line))
            test_y.append(test_label[i+n])
        for i in range(batch_size-(total_n_test-n)):
            rng = random.randint(0, len(test_text) - 1)
            line = test_text[rng]
            test_x.append(encode(vocabdict, line))
            test_y.append(test_label[rng])
        return np.array(test_x,dtype='float32'),np.array(test_y,dtype='int32'),n,None

def compute_accuracy(v_xs,v_ys):
    global result
    prediction=sess.run(result,feed_dict={xs:v_xs,ys:v_ys})
    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(v_ys,1))
    correct_prediction=sess.run(correct_prediction,feed_dict={xs:v_xs,ys:v_ys})
    return correct_prediction

def test_compute_accuracy(batch_size,test_text,test_label,vocabdict,istest):
    if istest:
        ckpt = tf.train.get_checkpoint_state(modelpath + 'bestmodel/')
        if ckpt:
            print('havebestmodel')
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, modelpath + "bestmodel/model.ckpt")
    n=0
    total_n_test=len(test_label)
    correct_prediction=[]
    while n<=total_n_test:
        t_xs,t_ys,n,tn=load_test_data(batch_size,test_text,test_label,vocabdict,n)
        if tn is not None:
            correct_prediction.extend(compute_accuracy(t_xs,t_ys))
        else:
            correct_prediction.extend(compute_accuracy(t_xs,t_ys)[:total_n_test-n])
            break
    accuracy =[float(i) for i in correct_prediction]
    accuracy=sum(accuracy)/len(accuracy)
    return accuracy


class LSTM():
    def __init__(self,xs,vocabdict,embedding_size,train_text,n_hidden_units,batch_size,text_len):
        embeddings = load_word_embeddings(vocabdict,embedding_size,train_text)

        x_in = tf.nn.embedding_lookup(embeddings,xs)                                                                        #按照随机提取的句子的编号，再次给emb排序
        x_in = tf.reshape(x_in, [-1, embedding_size])                                                                       #变成默认行，每行embedding_size个数据
        W = tf.Variable(tf.truncated_normal([embedding_size, n_hidden_units], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[n_hidden_units]))                                                           #全为0.1的1行n_hidden_units列数组
        x_in = tf.matmul(x_in, W) + b                                                                                       #n_hidden_units列
        x_in = tf.reshape(x_in, [-1, text_len, n_hidden_units])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
        _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)
        output = tf.concat(outputs, 2)
        x_lstm_out = tf.reshape(output, [-1, n_hidden_units])
        W_lstm_out = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, 1]))
        b_lstm_out = tf.Variable(tf.constant(0.1, shape=[1]))
        result = tf.matmul(x_lstm_out, W_lstm_out) + b_lstm_out
        result = tf.reshape(result, [batch_size, text_len, 1])
        self.result = tf.reshape(result, [batch_size, text_len])



def train(batch_size,train_text,train_label,vocabdict):
    max_accuracy=0
    for i in range(10000):
        train_x, train_y = load_train_data(batch_size, train_text, train_label,vocabdict)
        sess.run(trainstep,feed_dict={xs: train_x,ys: train_y})
        stracc=sess.run(accuracy,feed_dict={xs:train_x,ys:train_y})
        strtrain='train accuracy: %5f'%(stracc)
        print(strtrain)

        if i%100==0:
            saver.save(sess, modelpath+"model/model.ckpt")
            dev_accuracy = test_compute_accuracy(batch_size,dev_text,dev_label,vocabdict,False)
            strdev = 'epoch:' + str(i) + '   dev_accuracy: %5f' % (dev_accuracy)
            print(strdev)
            resultfile.write(strdev + '\r\n')
            if dev_accuracy >= max_accuracy:
                saver.save(sess,modelpath+"bestmodel/model.ckpt")
                max_accuracy = dev_accuracy

    test_accuracy=test_compute_accuracy(batch_size,test_text,test_label,vocabdict,True)
    strtest='test_accuracy: %f'%test_accuracy
    print(strtest)
    resultfile.write(strtest)


text_len = 52
n_hidden_units = 128
batch_size = 32
lr = 0.001
embedding_size = 400



train_text,train_label = load_corpus(traindir,text_len)
dev_text,dev_label = load_corpus(devdir,text_len)
test_text, test_label = load_corpus(testdir,text_len)
vocabdict=build_vocab(train_text)

xs=tf.placeholder(tf.int32,[None,None])
ys=tf.placeholder(tf.int32,[None,None])

model=LSTM(xs,vocabdict,embedding_size,train_text,n_hidden_units,batch_size,text_len)
result=model.result
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(ys,tf.float32),logits=result))
trainstep=tf.train.AdamOptimizer(lr).minimize(cost)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(result,1),tf.argmax(ys,1)),tf.float32))

sess=tf.Session(config=tf_config)
saver=tf.train.Saver()

ckpt=tf.train.get_checkpoint_state(modelpath+"model/")
if ckpt:
    print('havemodel')
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess,modelpath+"model/model.ckpt")
else:
    init = tf.global_variables_initializer()
    sess.run(init)

train(batch_size,train_text,train_label,vocabdict)