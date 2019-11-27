import cv2


class Net:
    def __init__(self, weights, config, cocoNames) -> object:
        self.weights = weights
        self.config = config
        self.names = cocoNames
        self.net = self.__getNet()
        self.classes = self.__getClasses()
        self.__layers = self.output_layers()
        self.layer_names = self.getLayerNames()

    def __getNet(self) -> object:
        return cv2.dnn.readNet(self.weights, self.config)

    def __getClasses(self):
        with open(self.names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes

    def getLayerNames(self):
        return self.net.getLayerNames()

    def output_layers(self):
        output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def set_input(self, blob):
        x = self.net.setInput(blob)

    def get_outs(self, blob):
        self.net.setInput(blob)
        outs = self.net.forward(self.__layers)
        return outs
