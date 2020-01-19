import os
import pandas as pd
import numpy as np
import tensorflow as tf



output_path = '/Users/Soohyun/PycharmProjects/MissingPredict/output_data/'
#input_data = np.loadtxt('input_path' + 'in_2015.csv', delimiter=',', dtype=np.float32)
#output_data = np.loadtxt('output_path' + 'out_2015.csv', delimiter=',', dtype=np.float32)

#input_data = pd.read_csv(input_path +'in_2015.csv')
#input_data1 = tf.read_file(input_path)
#output_data = pd.read_csv(output_path +'out_2015.csv')
#output_data1 = tf.read_file(output_path)
#merged_data = pd.merge(input_data, output_data)
#x_data = training_data[:, 0:4]
#y_data = training_data[:, 5:7]
def csv2list(filename):
    input_path = '/Users/Soohyun/PycharmProjects/MissingPredict/input_data/'
    lists = []
    file = open(input_path+filename, 'r', encoding='UTF8')
    while True:
        line = file.readline().rstrip("\n")
        if line:
            line = line.split(",")
            lists.append(line)
        else:
            break
    return lists

training_data_list = csv2list("in_2015.csv")

training_data = np.array(training_data_list)
x_training = training_data[0:, :5]
y_training = training_data[0:, 5:]
xs = x_training.tolist()
ys = y_training.tolist()


# DNN 모델과 최적화 함수에 따라 변경 필요.
# cost 값이 작아지면 learning rate 도 작게 변경해보면 좋음.
# 0.1 -> 0.01 -> 0.001
learning_rate = 0.1
# 15 -> 30
training_cnt = 200
batch_size = 150

# data set
X = tf.placeholder(tf.float32, [None, 5])
Y = tf.placeholder(tf.float32, [None, 3])

# drop out 에 사용할 값을 담을 변수정의
# 학습할 때는 0.5 ~ 0.7 정도의 뉴런을 활성화
# 테스트나 상용에서는 1의 뉴런을 활성화 하도록 설정할 예정.
keep_prob = tf.placeholder(tf.float32)

# DNN 모델로 변경
# L2 ~ L4 가 Hidden Layer --> hidden layer 2개로 변경
# 중간단계 shape 는 512 (반복 테스트를 통해 튜닝가능)

# 정확도를 높이기 위해 Xavier Initializer 사용하여 가중치를 초기화했다가,
# 자비에 함수는 양끝이 수렴하는 경우 적절하므로,
# ReLU 활성화 함수에 적합한 He 함수를 사용하도록 initializer 부분을 추가해줌.
# 이를 위해 get_variable() 함수를 사용.
W1 = tf.get_variable("W1", shape=[5, 4],
                     initializer=tf.keras.initializers.he_normal())
b1 = tf.Variable(tf.random_normal([4]))
# 활성화 함수로 ReLU 사용.
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# over fitting 이 일어나지 않도록 중간중간 무작위로 뉴런을 비활성화하여 조금더 일반화 시킴.
# 트레이닝 데이터셋의 정확도는 다소 떨어지더라도
# 새로운 데이터에 대한 정확도는 상승.
# tf.layer.dropout() 함수 사용법도 알아보시길..
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[4, 4],
                     initializer=tf.keras.initializers.he_normal())
b2 = tf.Variable(tf.random_normal([4]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.get_variable("W3", shape=[4, 3],
                     initializer=tf.keras.initializers.he_normal())
b3 = tf.Variable(tf.random_normal([3]))

"""
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[3, 3],
                     initializer=tf.keras.initializers.he_normal())
b4 = tf.Variable(tf.random_normal([3]))


L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.keras.initializers.he_normal())
b5 = tf.Variable(tf.random_normal([10]))
"""

logits = tf.matmul(L2, W3) + b3
cost = tf.reduce_mean(tf.square(logits -Y))
# 현재까지 연구된 최적화 함수중 가장 성능이 좋다고 평가됨.
# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
op_train = optimizer.minimize(cost)

pred = tf.nn.softmax(logits)
prediction = tf.argmax(pred, 1)
true_Y = tf.argmax(Y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, true_Y), dtype=tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_cnt):
    avg_cost = 0
    batch_cnt = int(len(training_data)/ batch_size)

    for i in range(batch_cnt):
        i = 1
        batch_xs = xs[i*batch_size:(i+1)*batch_size]
        batch_ys = ys[i*batch_size:(i+1)*batch_size]
        # 학습 구간에서는 drop out 에 사용될 값을 0.7로 설정함.
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.8}
        c,l, _ = sess.run([cost, logits, op_train], feed_dict=feed_dict)
        avg_cost += c / batch_cnt

    if ((epoch+1) == 1 or (epoch+1) % 10 == 0) or (epoch+1) == training_cnt:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')


"""
# 테스트 구간에서는 drop out 에 사용될 값을 1로 설정함.
print('Accuracy(train):', sess.run(accuracy, feed_dict={
      X: batch_xs, Y: batch_ys, keep_prob: 1.0}))
#
print('Accuracy(test):', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    prediction, feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1.0}))
"""


