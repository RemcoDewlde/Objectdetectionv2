import numpy as np


class Color:
    """
     A class used to represent a Color
    """

    def __init__(self, classes: list, random: bool):
        """
        Parameters
        ----------
        classes : List/Array
                List/Array of all the classes in coco.names dataset

        random : bool
                Generate a new array of colors or read settings from file

        """

        self._random = random
        self.classes = classes
        self.color = self.color()
        self.color_array = self.color

    def set_color(self):
        """ Sets / generates colors for every item in the self.classes array/list

        :returns colors array
        """
        if self._random:
            colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            with open('cfg/colors.txt', 'w') as file:
                file.writelines(["%s\n" % color for color in colors])
            return colors
        else:
            colors = []
            with open("cfg/colors.txt", "r") as c:
                for line in c:
                    colors = [line.strip() for line in c.readlines()]

                colors = np.asarray(colors)
                self.color_array = colors
                return colors

    def get_color(self):
        """ get colors for every item of the self.classes array/list
        :returns Color
        """
        return self.color
