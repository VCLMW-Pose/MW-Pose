import tensorflow as tf
from tensorflow.contrib import rnn

class BiRNNModel(object):
    def __init__(self, size, args):
        self.num_layers = args.num_layers#网络层数
        self.num_hidden = args.num_hidden#隐藏层层数

        self.x = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]

        self.lm_input = self.x#(输入矩阵)
        self.lm_output = self.x[:, 1:-1]#预期输出
        self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)#得到每一个张量（序列）的长度，lm_input大于零则sign为1，否则为-1

        with tf.name_scope("bi-rnn"):#多层双向RNN网络搭建
            def make_cell():#定义基本单元
                cell = rnn.BasicLSTMCell(self.num_hidden)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)#dropout操作，对于rnn的部分不进行dropout，也就是说从t-1时候的状态传递到t时刻进行计算时，这个中间不进行memory的dropout；仅在同一个t时刻中，多层cell之间传递信息的时候进行dropout
                return cell

            fw_cell = rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])#前向传播单元
            bw_cell = rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])#反向传播单元
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, lm_input, sequence_length=self.seq_len, dtype=tf.float32)

            fw_outputs = rnn_outputs[0][:, :-2, :]
            bw_outputs = rnn_outputs[1][:, 2:, :]
            merged_output = tf.concat([fw_outputs, bw_outputs], axis=2)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(merged_output, size)

        with tf.name_scope("loss"):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.lm_output,
                weights=tf.sequence_mask(self.seq_len - 2, tf.shape(self.x)[1] - 2, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True
            )
