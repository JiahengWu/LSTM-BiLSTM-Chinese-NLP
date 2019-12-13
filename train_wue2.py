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

# traindir='/home/lisi/wjh/wue/data/train.txt'
# devdir='/home/lisi/wjh/wue/data/dev.txt'
# testdir='/home/lisi/wjh/wue/data/test.txt'
# resultdir='/home/lisi/wjh/wue/result/'
# Word2Vecdir='/home/lisi/wjh/wue/data/wiki.zh.text.model'
# modelckptdir='/home/lisi/wjh/wue/result/modelckpt/'
# bestmodelckptdir='/home/lisi/wjh/wue/result/bestmodelckpt/'


traindir='E:\\python_project\\Chinese_wue\\data\\fivesec.txt'
devdir='E:\\python_project\\Chinese_wue\\data\\fivesec.txt'
testdir='E:\\python_project\\Chinese_wue\\data\\test.txt'
resultdir='E:\\python_project\\Chinese_wue\\result\\'
Word2Vecdir='E:\\python_project\\Chinese_wue\\data\\wiki.zh.text.model'
modelckptdir='E:\\python_project\\Chinese_wue\\result\\modelckpt\\'
bestmodelckptdir='E:\\python_project\\Chinese_wue\\result\\bestmodelckpt\\'


resultfile=codecs.open(resultdir+'/result.txt','a',encoding='utf-8')


#读取train文件 返回为 train_text和train_label，两个二维数组，错误处标记1，正确为0
def load_train_text(traindir,text_len):
    train_text = []
    train_label= []
    a = re.compile(u'[<][\u4e00-\u9fa5]+[>]')
    lines = codecs.open(traindir,'r', encoding='UTF-8').readlines()
    for line in lines:
        train_text_line = []
        train_label_line = []
        line = line.replace('\n', '').split(' ')
        # line = line.encode('utf-8').decode('utf-8-sig')
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
                train_text_line.append('<a>')                                                                           #每行剩下的用<a>来补
                train_label_line.append(0)

        train_text.append(train_text_line)
        train_label.append(train_label_line)

    return train_text,train_label

#提取出batch_size句话，每句话的标号放在train_x里，其对应 label放进train_y里
def load_train_data(batch_size,train_text,train_label):
    train_x=[]
    train_y=[]
    x_temp = []
    for i in range(0,batch_size):                                                                                      #提取出batch_size句话
        rng = random.randint(0,len(train_text)-1)                                                                       #随机找一行    e.g.120  0~300
        x_temp.append(rng)
        train_y.append(train_label[rng])
    train_x.append(x_temp)                                                                                              #train_x添加某句话在train_text中的编号
    return np.array(train_x,dtype='float32'),np.array(train_y,dtype='int32')

#把train_text里所有的词变成词向量放进embeddings里, embeddings三维[len(train_text)--batch,text_len,embedding_size]
def load_word_embeddings(embedding_size,train_text):      #-------------vector文件
    embeddings = [[[0.01 for i in range(embedding_size)] for j in range(text_len)]for k in range(len(train_text))]
    model = gensim.models.Word2Vec.load(Word2Vecdir)
    i=0
    for text in train_text:
        j = 0
        for word in text:
            if word in model:
                embeddings[i][j] = model[word]
                j+=1
            else:
                j+=1
        i+=1
    return tf.Variable(np.array(embeddings,dtype='float32'),dtype=tf.float32)

### test 部分 ###
def load_test_data(batch_size,test_text,test_label,n):
    test_x=[]
    test_y=[]
    test_x_temp = []
    total_n_test=len(test_label)
    if n+batch_size<total_n_test:
        for i in range(batch_size):
            test_x_temp.append(i+n)
            test_y.append(test_label[i+n])
        test_x.append(test_x_temp)
        n=n+batch_size
        return np.array(test_x,dtype='float32'),np.array(test_y,dtype='int32'),n,n
    else:
        for i in range(total_n_test-n):
            test_x_temp.append(i + n)
            test_y.append(test_label[i+n])
        for i in range(batch_size-(total_n_test-n)):
            rng = random.randint(0, len(test_text) - 1)
            test_x_temp.append(rng)
            test_y.append(test_label[rng])
        test_x.append(test_x_temp)
        return np.array(test_x,dtype='float32'),np.array(test_y,dtype='int32'),n,None

def compute_accuracy(v_xs,v_ys):
    global result
    prediction=sess.run(result,feed_dict={xs:v_xs,ys:v_ys})
    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(v_ys,1))
    correct_prediction=sess.run(correct_prediction,feed_dict={xs:v_xs,ys:v_ys})
    return correct_prediction

def test_compute_accuracy(test_text,test_label,istest):
    if istest:
        ckpt = tf.train.get_checkpoint_state(bestmodelckptdir)
        if ckpt:
            print('havebest')
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, bestmodelckptdir + 'bestmodel.ckpt')
    n=0
    total_n_test=len(test_label)
    correct_prediction=[]
    while n<=total_n_test:
        t_xs,t_ys,n,tn=load_test_data(batch_size,test_text,test_label,n)
        if tn is not None:
            correct_prediction.extend(compute_accuracy(t_xs,t_ys))
        else:
            correct_prediction.extend(compute_accuracy(t_xs,t_ys)[:total_n_test-n])
            break
    accuracy =[float(i) for i in correct_prediction]
    accuracy=sum(accuracy)/len(accuracy)
    return accuracy

class BiLSTM_Model():
    def __init__(self,xs,embedding_size,text_len,n_hidden_units,train_text):
        embeddings=load_word_embeddings(embedding_size, train_text)

        x_in=tf.nn.embedding_lookup(embeddings,xs)                                                                          #按照随机提取的句子的编号，再次给emb排序
        x_in = tf.reshape(x_in, [-1, embedding_size])                                                                       #变成默认行，每行embedding_size个数据
        W = tf.Variable(tf.truncated_normal([embedding_size, n_hidden_units], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[n_hidden_units]))                                                           #全为0.1的1行n_hidden_units列数组
        x_in = tf.matmul(x_in, W) + b                                                                                       #n_hidden_units列
        x_in = tf.reshape(x_in, [-1, text_len, n_hidden_units])

        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x_in, dtype=tf.float32)
        output=tf.concat(outputs,2)
        x_lstm_out = tf.reshape(output, [-1, n_hidden_units * 2])
        W_lstm_out = tf.Variable(tf.constant(0.1, shape=[n_hidden_units * 2, 1]))
        b_lstm_out = tf.Variable(tf.constant(0.1, shape=[1]))
        result = tf.matmul(x_lstm_out, W_lstm_out) + b_lstm_out
        result = tf.reshape(result, [batch_size, text_len, 1])
        self.result=tf.reshape(result,[batch_size,text_len])

def train(batch_size,train_text, train_label):
    max_accuracy=0
    for i in range(1000):
        train_x, train_y = load_train_data(batch_size, train_text, train_label)
        sess.run(trainstep,feed_dict={xs: train_x,ys: train_y})
        stracc=sess.run(accuracy,feed_dict={xs:train_x,ys:train_y})
        strtrain='train accuracy: %5f'%(stracc)
        print(strtrain)

        if i%100==0:
            saver.save(sess, modelckptdir+"model.ckpt")
            dev_accuracy = test_compute_accuracy(dev_text,dev_label,False)
            strdev = 'epoch:' + str(i) + '   dev_accuracy: %5f' % (dev_accuracy)
            print(strdev)
            resultfile.write(strdev + '\r\n')
            if dev_accuracy >= max_accuracy:
                saver.save(sess,bestmodelckptdir+"bestmodel.ckpt")
                max_accuracy = dev_accuracy

    test_accuracy=test_compute_accuracy(test_text,test_label,True)
    strtest='test_accuracy: %f'%test_accuracy
    print(strtest)
    resultfile.write(strtest)


lr=0.01
batch_size = 4
text_len = 10
embedding_size = 300
n_hidden_units = 128


train_text,train_label = load_train_text(traindir, text_len)
dev_text,dev_label = load_train_text(devdir,text_len)
test_text, test_label = load_train_text(traindir, text_len)

xs=tf.placeholder(tf.int32,[None,batch_size])
ys=tf.placeholder(tf.int32,[None,text_len])

pred=BiLSTM_Model(xs,embedding_size,text_len,n_hidden_units,train_text)
result=pred.result
print(result)
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(ys,tf.float32),logits=result))
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(result,1),tf.argmax(ys,1)),tf.float32))
trainstep=tf.train.AdamOptimizer(lr).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()


ckpt=tf.train.get_checkpoint_state(modelckptdir)
if ckpt:
    print('havemodel')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,modelckptdir+"model.ckpt")
else:
    sess.run(tf.global_variables_initializer())

train(batch_size,train_text, train_label)