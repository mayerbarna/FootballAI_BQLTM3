from keras.callbacks import TensorBoard


class TensorBoardConf:
    def __init__(self, log_dir: str = 'tracking/tensorboard/loss', histogram_freq: int = 1):
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq

    def __call__(self):
        return TensorBoard(log_dir=self.log_dir, histogram_freq=self.histogram_freq)

    def change_log_dir(self, new_log_dir: str):
        self.log_dir = new_log_dir
        return TensorBoard(log_dir=self.log_dir, histogram_freq=self.histogram_freq)

    def change_histogram_freq(self, new_histogram_freq: int):
        self.histogram_freq = new_histogram_freq
        return TensorBoard(log_dir=self.log_dir, histogram_freq=self.histogram_freq)