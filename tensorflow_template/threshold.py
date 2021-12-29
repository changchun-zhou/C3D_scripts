import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense
graph = tf.Graph()
with graph.as_default():
    input_t = tf.placeholder(tf.float32, [2,3], 'input')
    threshold_t = tf.Variable(0.05)
    fc1= Dense(100, activation='relu')
    fc2= Dense(3, activation='softmax')
    out_t = fc2(fc1(tf.maximum(input_t, threshold_t)))
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        print('out', sess.run(out_t, feed_dict={input_t: [[-0.3, 0.0, 0.7],[-0.3, 0.0, 0.7]]}))

        # get grad of out_t wrt threshold_t
        grad_out_t = tf.gradients(out_t, [threshold_t])[0]
        print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.0, 0.7],[-0.3, 0.0, 0.7]]}))
        print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.0, -0.7],[-0.3, 0.0, 0.7]]}))
        print('d(out)/d(theshold)', sess.run(grad_out_t, feed_dict={input_t: [[-0.3, 0.5, 0.7],[-0.3, 0.0, 0.7]]}))