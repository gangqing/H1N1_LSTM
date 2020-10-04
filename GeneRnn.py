from tensorflow.compat import v1 as tf
from H1N1.GeneData import Data
import H1N1.framework as fm


class GeneConfig(fm.Config):
    def __init__(self):
        super().__init__()
        self.china_path = "data/GeneFastaResults_China_4_8_2020.fasta"
        self.canada_path = "data/GeneFastaResults_Canada_4_8_2020.fasta"
        self.australia_path = "data/GeneFastaResults_Australia_5_8_2020.fasta"
        self.hongkong_path_2009 = "data/GeneFastaResults_HongKong_2009.fasta"
        self.kenya_path_2009 = "data/GeneFastaResults_Kenya.fasta"
        self.egypt_2009_path = "data/GeneFastaResults_ Egypt_2009.fasta"
        self.guam_2009_path = "data/GeneFastaResults_Guam_2009.fasta"
        self.iran_2009_2010_path = "data/GeneFastaResults_lran_2009-2010.fasta"
        self.guam_2009_txt_path = "data/archaeopteryx_2009_Guam.txt"
        self.num_chars = 4
        self.num_units = 10
        self.ds = None
        self.batch_size = 1
        self.new_model = False
        self.epoches = 6000
        self.lr = 0.0001
        # todo
        self.simple_length = 33
        self.gene_length = 1701  # 一个样本的基因长度

    def get_name(self):
        return "gene"

    def get_sub_tensors(self, gpu_index):
        return SubTensors(self)

    def get_app(self):
        return GeneApp(self)

    def get_ds_train(self):
        self.read_ds()
        return self.ds

    def get_ds_test(self):
        self.read_ds()
        return self.ds

    def read_ds(self):
        if self.ds is None:
            self.ds = Data(self)


class SubTensors:
    def __init__(self, config: GeneConfig):
        self.config = config
        # 训练网络
        self.x = tf.placeholder(tf.int64, [None, config.gene_length], name="x")  # [-1, 1701]
        self.inputs = [self.x]
        x = tf.one_hot(self.x, config.num_chars)  # [-1, 1701, 4]
        y = tf.layers.dense(x, config.num_units, activation=tf.nn.relu, name="dense_1")  # [-1, 1701, num_units]

        cell1 = tf.nn.rnn_cell.LSTMCell(config.num_units, name="cell_1", state_is_tuple=False)
        cell2 = tf.nn.rnn_cell.LSTMCell(config.num_units, name="cell_2", state_is_tuple=False)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple=False)
        state = cell.zero_state(config.gene_length, dtype=y.dtype)  # [-1, num_units * 4]

        losses = []
        y_predicts = []  # [simple_length - 1, 1701]
        with tf.variable_scope("for") as scope:
            for i in range(config.simple_length):
                inputs = y[i, :, :]  # [1701, num_units]
                predict_y, state = cell(inputs, state)  # [1701, num_units]
                logits = tf.layers.dense(predict_y, config.num_chars, name="dense_2")  # [1701, 4]

                if i + 1 < config.simple_length:
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(x[i + 1, :, :], logits)  # [1701]
                    losses.append(loss)  # [simple_length - 1, 1701]
                    y_predict = tf.argmax(logits, axis=1)  # [1701]
                    y_predicts.append(y_predict)  # [simple_length - 1, 1701]
                scope.reuse_variables()
        self.losses = [tf.reduce_mean(losses)]
        # self.precise = self.get_precise(y_predicts)
        # self.predict = tf.transpose(y_predicts, [1, 0])  # [batch_size - 1, 1700]

        # 预测网络
        self.predict_x = tf.placeholder(tf.int64, [config.gene_length], name="predict_x")  # [1701]
        self.state = tf.placeholder(tf.float32, [config.gene_length, 4 * config.num_units], name="state")  # [1701, 4 * 200]
        self.zero_state = cell.zero_state(tf.shape(self.predict_x)[0], dtype=y.dtype)  # [1701, 4 * 200]
        start_x = tf.one_hot(self.predict_x, config.num_chars)  # [1701, 4]
        tf.get_variable_scope().reuse_variables()
        start_x = tf.layers.dense(start_x, config.num_units, name="dense_1")  # [1701, num_units]
        with tf.variable_scope("for"):
            yi, self.next_state = cell(start_x, self.state)  # y : [1701, num_units]
            yi = tf.layers.dense(yi, config.num_chars, name="dense_2")  # [1701, 4]
        self.yi_predicts = tf.argmax(yi, axis=1)  # [1701]

    def get_precise(self, y_predicts):
        """
        计算精度
        :param y_predicts: [simple_length - 1, 1701]
        :return:
        """
        x = self.inputs[0]  # [None, 1701]
        precises = []  # [simple_length - 1]
        for i in range(config.simple_length - 1):
            precise = tf.equal(x[i + 1, :], y_predicts[i, :])
            precises.append(tf.cast(precise, tf.float64))

        return tf.reduce_mean(precises)


class GeneApp(fm.App):
    def __init__(self, config: GeneConfig):
        super().__init__(config)

    # def after_epoch(self, epoch):
    #     super(GeneApp, self).after_epoch(epoch)
        # fd = self.get_feed_dict(self.config.get_ds_train())
        # precise = self.session.run(self.ts.sub_ts[-1].precise, fd)
        # print('Epoch %d: precise = %.6f' % (epoch, precise))

    def test(self, ds_test):

        ds = self.config.ds
        datas = ds.next_batch(1)[0]
        # x = datas[0]
        # y = datas[1]
        ts = self.ts.sub_ts[-1]
        # state = self.session.run(ts.zero_state, {ts.predict_x: x})
        # y_predict, _ = self.session.run([ts.yi_predicts, ts.next_state], {ts.predict_x: x, ts.state: state})
        # y_predict = ds.to_gene(*y_predict)
        # print("predict : {y}".format(y=y_predict))

        # x = datas[0]
        x = ds.dic["JF500448"]
        state = self.session.run(ts.zero_state, {ts.predict_x: x})
        # for i in range(len(datas) - 1):
        for i in range(1):
            y_predict, state = self.session.run([ts.yi_predicts, ts.next_state], {ts.predict_x: x, ts.state: state})
            x = datas[i + 1]
        y_pre = ds.to_gene(*y_predict)
        y = ds.to_gene(*datas[-1])
        print("x : {x}".format(x=ds.to_gene(*x)))
        print("predict : {y}".format(y=y_pre))
        e_count = 0
        for yi, pi in zip(y, y_pre):
            if yi == pi:
                e_count += 1
        print("precise : {p}".format(p=(e_count/len(y))))
        # precise = np.mean(np.cast(np.equal(y, y_pre)))
        # print("precise : {p}".format(p=precise))


if __name__ == '__main__':
    config = GeneConfig()
    config.call("test")
    # ds = config.get_ds_train().next_batch(1)[0]
    # print(ds)
