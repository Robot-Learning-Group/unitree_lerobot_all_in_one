import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from unitree_lerobot.eval_robot.configured_arm import ArmJointLayout
from unitree_lerobot.utils.constants import ROBOT_CONFIGS, G1_DEX1_CONFIG


class ArmLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.states = {'left_arm': {'qpos': list(np.arange(7) / 10)},
                       'right_arm': {'qpos': list(np.arange(7, 14) / 10)}}
        self.write_episode()
        self.cfg = SimpleNamespace(robot_type='G1_RIGHT_ARM_2CAM_CONFIG', arm='G1_29',
                                   use_dataset=False, visualization=False, frequency=30,
                                   initial_pose_episode=self.temp.name, image_port=55555,
                                   camera_ports={})

    def write_episode(self):
        (Path(self.temp.name) / 'data.json').write_text(json.dumps({'data': [{'states': self.states}]}))

    def test_right_only_holds_left_and_does_not_mutate_initial_pose(self):
        layout = ArmJointLayout(self.cfg)
        first = layout.initial_pose.copy()
        target = layout.arm_target(np.ones(7))
        np.testing.assert_array_equal(target[:7], first[:7])
        np.testing.assert_array_equal(target[7:], np.ones(7))
        np.testing.assert_array_equal(layout.initial_pose, first)
        self.assertEqual(layout.state_indices, list(range(7, 14)))

    def test_left_both_and_reordered_arms(self):
        for fields, indices in [(['left_arm.qpos'], list(range(7))),
                                (['left_arm.qpos', 'right_arm.qpos'], list(range(14))),
                                (['right_arm.qpos', 'left_arm.qpos'], list(range(7, 14)) + list(range(7)))]:
            with self.subTest(fields=fields):
                base = ROBOT_CONFIGS[self.cfg.robot_type]
                config = replace(base, motors=[G1_DEX1_CONFIG.motors[i] for i in indices],
                                 json_state_data_name=fields, json_action_data_name=fields)
                with patch.dict(ROBOT_CONFIGS, {'test_layout': config}):
                    cfg = copy.copy(self.cfg); cfg.robot_type = 'test_layout'
                    layout = ArmJointLayout(cfg)
                    action = np.arange(len(indices)) / 20
                    target = layout.arm_target(action)
                    np.testing.assert_array_equal(target[indices], action)
                    self.assertEqual(layout.state_indices, indices)
                    unused = sorted(set(range(14)) - set(indices))
                    np.testing.assert_array_equal(target[unused], layout.initial_pose[unused])

    def test_camera_selection_and_port_override(self):
        base = ROBOT_CONFIGS[self.cfg.robot_type]
        config = replace(base, cameras=['left'], camera_to_image_key={'color_1': 'left'})
        with patch.dict(ROBOT_CONFIGS, {'test_layout': config}):
            self.cfg.robot_type = 'test_layout'; self.cfg.camera_ports = {'color_1': 60010}
            self.assertEqual(ArmJointLayout(self.cfg).camera_ports, {'left': 60010})

    def test_invalid_action(self):
        layout = ArmJointLayout(self.cfg)
        for action in [np.zeros(14), np.zeros((1, 7)), [float('nan')] * 7]:
            with self.assertRaises(ValueError): layout.arm_target(action)

    def test_invalid_episode_pose(self):
        self.states['left_arm']['qpos'] = [float('nan')] * 7
        self.write_episode()
        with self.assertRaises(ValueError): ArmJointLayout(self.cfg)

    def test_unsupported_channels_and_wrong_joint_order(self):
        base = ROBOT_CONFIGS[self.cfg.robot_type]
        for config in [replace(base, json_action_data_name=['right_arm.target_pos']),
                       replace(base, motors=list(reversed(base.motors)))]:
            with patch.dict(ROBOT_CONFIGS, {'test_layout': config}):
                self.cfg.robot_type = 'test_layout'
                with self.assertRaises(ValueError): ArmJointLayout(self.cfg)

    def test_model_and_dataset_validation(self):
        layout = ArmJointLayout(self.cfg)
        features = {'observation.state': {'shape': [7]}, 'action': {'shape': [7]}}
        features.update({f'observation.images.{name}': {'shape': [480, 640, 3]} for name in layout.config.cameras})
        policy = SimpleNamespace(input_features={key: SimpleNamespace(shape=value['shape']) for key, value in features.items() if key != 'action'},
                                 output_features={'action': SimpleNamespace(shape=[7])})
        layout.validate_features(features, policy)
        features['action']['names'] = list(reversed(layout.config.motors))
        with self.assertRaises(ValueError): layout.validate_features(features, policy)
        features['action']['names'] = [layout.config.motors]
        layout.validate_features(features, policy)
        policy.output_features['action'].shape = [14]
        with self.assertRaises(ValueError): layout.validate_features(features, policy)
        policy.output_features['action'].shape = [7]
        del features['observation.images.cam_right_wrist']
        with self.assertRaises(ValueError): layout.validate_features(features, policy)

    def test_hand_types_and_partial_right_hand_mapping(self):
        from unitree_lerobot.utils.constants import G1_DEX3_CONFIG, G1_BRAINCO_CONFIG, G1_INSPIRE_CONFIG
        for ee, base in [('dex1', G1_DEX1_CONFIG), ('dex3', G1_DEX3_CONFIG),
                         ('brainco', G1_BRAINCO_CONFIG), ('inspire1', G1_INSPIRE_CONFIG)]:
            with self.subTest(ee=ee):
                dof = (len(base.motors) - 14) // 2
                self.states['left_ee'] = {'qpos': [0.15] * dof}
                self.states['right_ee'] = {'qpos': [0.25] * dof}
                self.write_episode()
                config = replace(ROBOT_CONFIGS['G1_RIGHT_ARM_2CAM_CONFIG'],
                                 motors=base.motors[7:14] + base.motors[14+dof:],
                                 json_state_data_name=['right_arm.qpos', 'right_ee.qpos'],
                                 json_action_data_name=['right_arm.qpos', 'right_ee.qpos'])
                with patch.dict(ROBOT_CONFIGS, {'with_hand': config}):
                    self.cfg.robot_type = 'with_hand'; self.cfg.ee = ee
                    layout = ArmJointLayout(self.cfg)
                    self.assertEqual(layout.ee_dof, dof)
                    target = layout.full_target([0.4] * (7+dof))
                    np.testing.assert_allclose(target[:7], layout.initial_pose[:7])
                    np.testing.assert_allclose(target[7:14], [0.4]*7)
                    np.testing.assert_allclose(target[14:14+dof], [0.15]*dof)
                    np.testing.assert_allclose(target[14+dof:], [0.4]*dof)
                    state = layout.observation_state(np.arange(14), np.arange(2*dof))
                    np.testing.assert_array_equal(state, list(range(7,14)) + list(range(dof,2*dof)))
                    self.cfg.ee = 'invalid'
                    with self.assertRaises(ValueError): ArmJointLayout(self.cfg)

    def test_hand_shared_memory_scalar_and_array(self):
        from multiprocessing import Array, Value, Lock
        base = ROBOT_CONFIGS['G1_RIGHT_ARM_2CAM_CONFIG']
        for ee, dof, robot in [('dex1',1,G1_DEX1_CONFIG), ('dex3',7,ROBOT_CONFIGS['Unitree_G1_Dex3'])]:
            self.states.update(left_ee={'qpos':[0.1]*dof}, right_ee={'qpos':[0.2]*dof})
            self.write_episode()
            config = replace(base, motors=robot.motors,
                             json_state_data_name=['left_arm.qpos','right_arm.qpos','left_ee.qpos','right_ee.qpos'],
                             json_action_data_name=['left_arm.qpos','right_arm.qpos','left_ee.qpos','right_ee.qpos'])
            with patch.dict(ROBOT_CONFIGS, {'with_hand':config}):
                self.cfg.robot_type='with_hand'; self.cfg.ee=ee
                layout=ArmJointLayout(self.cfg)
                shared={'lock':Lock(),'state':Array('d',[0.3]*(2*dof)),
                        'left':Value('d',0) if dof==1 else Array('d',dof),
                        'right':Value('d',0) if dof==1 else Array('d',dof)}
                layout.command_hands(shared,[0.5]*dof+[0.6]*dof)
                for side,value in [('left',0.5),('right',0.6)]:
                    actual=[shared[side].value] if dof==1 else list(shared[side][:])
                    np.testing.assert_allclose(actual,[value]*dof)
                np.testing.assert_allclose(layout.read_hands(shared),[0.3]*(2*dof))


if __name__ == '__main__':
    unittest.main()
