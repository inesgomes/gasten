class Metric:
    def __init__(self):
        self.result = float('inf')

    def update(self, images):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def get_result(self):
        return self.result

    def reset(self):
        raise NotImplementedError
