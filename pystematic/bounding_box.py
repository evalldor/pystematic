import numpy as np

class BoundingBox:
    
    def __init__(self, *coords, mode="XYXY", **kwargs):

        #Allow coordinates to be passed as both a sequence and as individual arguments
        if len(coords) == 1:
            coords = coords[0]
        
        if mode == "XYXY":
            self._data = np.array(coords).reshape(-1)
        else:
            raise Exception(f"Unkown mode '{mode}'.")

        assert len(self._data) == 4

    @property
    def xmin(self):
        return self._data[0]
        
    @xmin.setter
    def xmin(self, value):
        self._data[0] = value

    @property
    def ymin(self):
        return self._data[1]

    @ymin.setter
    def ymin(self, value):
        self._data[1] = value

    @property
    def xmax(self):
        return self._data[2]

    @xmax.setter
    def xmax(self, value):
        self._data[2] = value

    @property
    def ymax(self):
        return self._data[3]

    @ymax.setter
    def ymax(self, value):
        self._data[3] = value

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin
    
    @property
    def area(self):
        return self.width * self.height
    
    def intersection_area(self, other):
        xmin = np.maximum(self.xmin, other.xmin)
        ymin = np.maximum(self.ymin, other.ymin)

        xmax = np.maximum(xmin, np.minimum(self.xmax, other.xmax))
        ymax = np.maximum(ymin, np.minimum(self.ymax, other.ymax))

        return BoundingBox(xmin, ymin, xmax, ymax).area

    def union_area(self, other):
        return self.area + other.area - self.intersection_area(other)

    def iou(self, other):
        intersection = self.intersection_area(other)
        
        return intersection / (self.area + other.area - intersection)

    def numpy(self, mode="XYXY"):
        if mode == "XYXY":
            return np.array(self._data)
        
        raise Exception(f"Unkown mode '{mode}'.")
    
    def array(self, mode="XYXY"):
        if mode == "XYXY":
            return [float(x) for x in self._data]
        
        raise Exception(f"Unkown mode '{mode}'.")

    def __str__(self):
        return f"bbox({str(self.array())})"

    def __repr__(self):
        return str(self.array())

    def serialize(self):
        return self.array()