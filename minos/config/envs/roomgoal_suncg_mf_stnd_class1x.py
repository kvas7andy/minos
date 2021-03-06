from minos.lib.util.measures import MeasureGoalRoomType

config = {
    'task': 'room_goal',
    'goal': {'minRooms': 1, 'roomTypes': 'any', 'select': 'random'},
    'observations': {'color': True, 'forces': False, 'audio': False, 'objectType': True, 'depth': False},
    'frame_skip': 1,
    'measure_fun': MeasureGoalRoomType(),
    'reward_type': 'dist_time',
    'agent': {'radialClearance': 0.2},
    'scenes_file': '../data/scenes.multiroom.csv',
    'states_file': '../data/episode_states.suncg.csv.bz2',
    'scene': {'arch_only': False, 'retexture': True, 'empty_room': False, 'dataset': 'p5dScene'},
    'scene_filter': lambda s: 2 < s['nrooms'] < 6,
    'episode_filter': lambda e: e['pathNumDoors'] > 1,
    'objective_size': 9, # For UNREAL
    'objecttypes_file': '../../objectTypes_1x.csv',
    'width': 88,
    'height': 88
}
