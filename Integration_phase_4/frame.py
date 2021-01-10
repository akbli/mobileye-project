class Frame:
    def __init__(self):
        self.frame_id = 0
        self.path_frame = ""
        self.light_candidates = []
        self.tfl_candidates = []
        self.light_auxiliary = []
        self.tfl_auxiliary = []
        self.tfl_distance = []
        self.img = None
