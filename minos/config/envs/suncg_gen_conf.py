from minos.lib.util.measures import MeasureDistDirTime

config = {
    'task': 'point_goal',
    'goal': {'position': 'random', 'radius': 0.25},
    'measure_fun': MeasureDistDirTime(goal_dist_threshold=0.4),
    'reward_type': 'dist_time',
    'agent': {'radialClearance': 0.2},
    'scenes_file': '../data/scenes.multiroom.csv',
    'states_file': '../data/episode_states.suncg.csv.bz2',
    'scene': {'arch_only': False, 'retexture': True, 'empty_room': False, 'dataset': 'p5dScene'},
    'scene_filter': lambda s: 2 < s['nrooms'] < 6 and s['id'] != '58306be361138410c5383bfc6af2b843',
    'episode_filter': lambda e: e['pathNumDoors'] > 1,
    'objective_size': 4 # For UNREAL
}
