from network import Network

class MNISTAdverserial(Network):
    def setup_with_dropout(self,factor):
        (self.feed('data')
             .reshape([-1,28,28,1], name='data_reshape')
             .conv(5, 5, 20, 1, 1, padding='VALID', relu=False, name='conv1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(5, 5, 50, 1, 1, padding='VALID', relu=False, name='conv2')
             .max_pool(2, 2, 2, 2, name='feat'))
        
        (self.feed('feat')
             .fc(500, name='fc6', with_bn=True)
             .fc(10, relu=False, name='fc7'))
        
        (self.feed('feat')
            .flip_grad(name='flip_grad',l=factor)
            .fc(512, name='adv_fc2',with_bn=True)
            .fc(2, relu=False, name='adv_fc3'))

