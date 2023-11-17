
params = {
    'world_size' : [10,10,10], #meters
    'scale' : 0.10,    # 1 voxel = s meters 
    'robot_bounding_sphere_diameter' : 0.5, #meters 
    'purr_threshold' : 10000,
    'start' : [38,59,42],
    'end' : [60,56,49]
}

class Config:
    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            value = self.params[key]
            if isinstance(value, dict):
                return Config(value)  # Recursively create Config object for nested dictionaries
            else:
                return value
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

cfg = Config(params)