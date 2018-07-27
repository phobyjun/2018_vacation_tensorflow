import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# a, b, c 상수 선언
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

# 이미 만든 두 개의 변수를 입력으로 사용해 연산 수행
d = tf.multiply(a, b)
e = tf.add(c, b)
f = tf.subtract(d, e)

# 세션 만들고 실행
sess = tf.Session()
outs = sess.run(f)
sess.close()
print("outs = {}".format(outs))
