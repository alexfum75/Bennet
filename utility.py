import yaml
import numpy as np

def findComet (cel_bodies):
    bodies = {}
    for body_index in range(0, len(cel_bodies['bodies'])):
        for _body, _param in cel_bodies['bodies'][body_index].items():
            if 'comet' not in _param:
                _param['comet'] = 'False'
            else:
                bodies[_body] = _body.lower()
    return bodies


def calculateAngle(coord_p_1, coord_p_2, coord_p_3):
    p_1 = np.array(coord_p_1)
    p_2 = np.array(coord_p_2)
    p_3 = np.array(coord_p_3)

    # calculate vectors
    vec_p1p2 = p_2 - p_1
    vec_p1p3 = p_3 - p_1
    vec_p2p1 = p_1 - p_2
    vec_p2p3 = p_3 - p_2
    vec_p3p1 = p_1 - p_3
    vec_p3p2 = p_2 - p_3

    # calculate vertex p1
    cos_p1 = np.dot(vec_p1p2, vec_p1p3) / (np.linalg.norm(vec_p1p2, ord=2) * np.linalg.norm(vec_p1p3, ord=2))
    p1_rad = np.arccos(cos_p1)
    p1_deg = np.degrees(p1_rad)

    # calculate vertex p2
    cos_p2 = np.dot(vec_p2p1, vec_p2p3) / (np.linalg.norm(vec_p2p1, ord=2) * np.linalg.norm(vec_p2p3, ord=2))
    p2_rad = np.arccos(cos_p2)
    p2_deg = np.degrees(p2_rad)

    # calculate vertex p3
    cos_p3 = np.dot(vec_p3p1, vec_p3p2) / (np.linalg.norm(vec_p3p1, ord=2) * np.linalg.norm(vec_p3p2, ord=2))
    p3_rad = np.arccos(cos_p3)
    p3_deg = np.degrees(p3_rad)
    return p1_deg, p2_deg, p3_deg


def readConfig(filename):
    _bodies = None
    with open(filename) as stream:
        try:
            _bodies = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return _bodies
