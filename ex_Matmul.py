import tensorflow as tf

A = tf.constant([[1, 2, 3],
                 [4, 5, 6]])

x = tf.constant([1, 0, 1])

# x에 차원 추가
x = tf.expand_dims(x, 1)
print(x.get_shape())

b = tf.matmul(A, x)

sess = tf.InteractiveSession()
print("matmul result:\n {}".format(b.eval()))
sess.close()
