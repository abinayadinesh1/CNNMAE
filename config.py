class Config():
    def __init__(self, channel_list, batch_size=1, input_dims=(256, 256)):
        self.channel_list = channel_list

        self.num_blocks = len(channel_list)-1

        self.batch_size = batch_size

        self.input_dims = input_dims