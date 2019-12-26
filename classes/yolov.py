import cv2


class Net:
    def __init__(self, weights, config, cocoNames) -> object:
        """
        :parameter
        ----------
        weights : yolov weights file
        config : yolov config file
        cocoNames : coco dataset class names

        """

        self.weights = weights
        self.config = config
        self.names = cocoNames
        self.net = self.__getNet()
        self.classes = self.__getClasses()
        self.__layer_names = self.getLayerNames()
        self.__output_layers = self.output_layers()

    def __getNet(self) -> object:
        """
        :return Net -> opencv2.dnn.readnet
        """
        return cv2.dnn.readNet(self.weights, self.config)

    def __getClasses(self):
        """
        :return array with class names
        """
        with open(self.names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
            return classes

    def getLayerNames(self):
        """"
        :return cv2.net.getLayerNames()
        """

        return self.net.getLayerNames()

    def output_layers(self):
        """
        returns the output laysers of the
        :return output_layers
        """
        output_layers = self.net.getUnconnectedOutLayers()
        return output_layers

    def get_outs(self, blob):
        """
        :param blob
        :return outs
        """
        self.net.setInput(blob)
        outs = self.net.forward(self.__output_layers)
        return outs
