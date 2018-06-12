from network import Network


class LeNet(Network):
    def setup(self):
        (self.feed('data')
             .reshape([-1,28,28,1], name='data_reshape')
             .conv(5, 5, 20, 1, 1, padding='VALID', relu=False, name='conv1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(5, 5, 50, 1, 1, padding='VALID', relu=False, name='conv2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .fc(500, name='ip1')
             .fc(10, relu=False, name='ip2')
             .softmax(name='prob'))

