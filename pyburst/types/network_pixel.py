class Pixel:
    __slots__ = ['pixel', 'rate']

    def __init__(self, pixel):
        #: ROOT.pixel object
        self.pixel = pixel
        self.rate = pixel.rate
