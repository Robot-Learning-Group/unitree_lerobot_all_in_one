"""Run inside the LeRobot environment; all robot interfaces are mocked."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
import cv2
import zmq
from unitree_lerobot.eval_robot.configured_arm import ArmJointLayout, eval_configured_arm
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / 'data.json').write_text(json.dumps({'data': [{'states': {
            'left_arm': {'qpos': [0.1]*7}, 'right_arm': {'qpos': [0.2]*7}}}]}))
        self.cfg = EvalRealConfig(repo_id='local/test', robot_type='G1_RIGHT_ARM_2CAM_CONFIG',
                                  initial_pose_episode=self.tmp.name, frequency=1)
        self.layout = ArmJointLayout(self.cfg)
        self.pose = np.zeros(14)
        self.ctrl = Mock()
        self.ctrl.get_current_dual_arm_q.side_effect = lambda: self.pose.copy()
        def command(target, tau): self.pose = target.copy()
        self.ctrl.ctrl_dual_arm.side_effect = command
        self.setup = Mock(return_value={'arm_ctrl':self.ctrl, 'arm_ik':Mock(solve_tau=lambda q: np.zeros(14))})
        robot_module = ModuleType('unitree_lerobot.eval_robot.make_robot')
        robot_module.setup_robot_interface = self.setup
        self.enter(patch.dict('sys.modules', {'unitree_lerobot.eval_robot.make_robot':robot_module}))
        frame = np.full((480,640,3), [10,20,30], dtype=np.uint8)
        jpg = cv2.imencode('.jpg',frame)[1].tobytes()
        self.socks = [Mock(), Mock()]
        for s in self.socks: s.recv.return_value=jpg
        self.context=Mock(); self.context.socket.side_effect=self.socks
        self.enter(patch('zmq.Context',return_value=self.context))
        self.enter(patch('builtins.input',return_value='s'))
        self.enter(patch('unitree_lerobot.eval_robot.configured_arm.time.sleep'))
        features={f'observation.images.{n}':{'shape':[480,640,3]} for n in self.layout.config.cameras}
        self.dataset=Mock(); self.dataset.meta=SimpleNamespace(features=features,episodes={'dataset_from_index':[0]})
        # MagicMock supplies indexed dataset access.
        from unittest.mock import MagicMock
        self.dataset=MagicMock(meta=self.dataset.meta); self.dataset.__getitem__.return_value={'task':'test'}
        self.policy=Mock(config=SimpleNamespace(device='cpu',use_amp=False))
        self.predict=self.enter(patch('unitree_lerobot.eval_robot.utils.utils.predict_action'))

    def enter(self, patcher):
        result=patcher.start(); self.addCleanup(patcher.stop); return result

    def run_loop(self):
        eval_configured_arm(self.cfg,self.dataset,self.policy,Mock(),Mock(),self.layout)

    def test_action_routing_camera_rgb_and_gripper_disabled(self):
        self.predict.side_effect=[torch.full((7,),0.4),KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt): self.run_loop()
        self.assertEqual(self.setup.call_args.args[0].ee,'')
        np.testing.assert_allclose(self.pose[:7],[0.1]*7)
        np.testing.assert_allclose(self.pose[7:],[0.4]*7)
        observation=self.predict.call_args_list[0].args[0]
        np.testing.assert_allclose(observation['observation.state'],[0.2]*7)
        self.assertEqual(set(observation),{'observation.state','observation.images.cam_high','observation.images.cam_right_wrist'})
        np.testing.assert_allclose(observation['observation.images.cam_high'][0,0],[30,20,10],atol=2)
        self.assertTrue(all(s.close.called for s in self.socks))

    def test_cancel_at_prompt_never_constructs_controller(self):
        with patch('builtins.input', return_value='q'):
            self.run_loop()
        self.setup.assert_not_called()

    def test_camera_failure_before_start_never_constructs_controller(self):
        self.socks[1].recv.side_effect=zmq.Again()
        with self.assertRaises(zmq.Again): self.run_loop()
        self.setup.assert_not_called()

    def test_invalid_action_holds_measured_pose(self):
        self.predict.return_value=torch.full((7,),float('nan'))
        with self.assertRaises(ValueError): self.run_loop()
        np.testing.assert_allclose(self.pose,self.layout.initial_pose)
        self.assertTrue(np.isfinite(self.pose).all())

    def test_right_hand_runtime_uses_selected_controller_and_holds_left(self):
        from dataclasses import replace
        from multiprocessing import Value, Array, Lock
        from unitree_lerobot.utils.constants import ROBOT_CONFIGS, G1_DEX1_CONFIG
        data={'data':[{'states':{'left_arm':{'qpos':[0.1]*7},'right_arm':{'qpos':[0.2]*7},
                                'left_ee':{'qpos':[0.15]},'right_ee':{'qpos':[0.25]}}}]}
        (Path(self.tmp.name)/'data.json').write_text(json.dumps(data))
        base=ROBOT_CONFIGS['G1_RIGHT_ARM_2CAM_CONFIG']
        config=replace(base,motors=G1_DEX1_CONFIG.motors[7:14]+G1_DEX1_CONFIG.motors[15:],
                       json_state_data_name=['right_arm.qpos','right_ee.qpos'],
                       json_action_data_name=['right_arm.qpos','right_ee.qpos'])
        shared={'lock':Lock(),'state':Array('d',[0.15,0.25]),'left':Value('d',0),'right':Value('d',0)}
        self.setup.return_value.update(ee_dof=1,ee_shared_mem=shared)
        self.cfg.robot_type='test_hand'; self.cfg.ee='dex1'
        with patch.dict(ROBOT_CONFIGS,{'test_hand':config}):
            self.layout=ArmJointLayout(self.cfg)
            def stop():
                self.assertAlmostEqual(shared['left'].value,0.15)
                self.assertAlmostEqual(shared['right'].value,0.6,places=6)
                raise KeyboardInterrupt()
            self.predict.side_effect=[torch.tensor([0.4]*7+[0.6])]
            calls=0
            def predict(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls>1: stop()
                np.testing.assert_allclose(args[0]['observation.state'],[0.2]*7+[0.25])
                return torch.tensor([0.4]*7+[0.6])
            self.predict.side_effect=predict
            with self.assertRaises(KeyboardInterrupt): self.run_loop()
        self.assertEqual(self.setup.call_args.args[0].ee,'dex1')
        np.testing.assert_allclose(self.pose[:7],[0.1]*7)
        self.assertAlmostEqual(shared['right'].value,0.25)  # hold measured hand state on exit

    def test_hand_not_at_initial_pose_prevents_inference(self):
        from multiprocessing import Value, Array, Lock
        self.layout.ee='dex1'; self.layout.ee_dof=1
        self.layout.initial_full_pose=np.concatenate((self.layout.initial_pose,[0.1,0.2]))
        shared={'lock':Lock(),'state':Array('d',[0.9,0.9]),'left':Value('d',0),'right':Value('d',0)}
        self.setup.return_value.update(ee_dof=1,ee_shared_mem=shared)
        with patch('unitree_lerobot.eval_robot.configured_arm.time.monotonic',side_effect=[0,11]):
            with self.assertRaises(TimeoutError): self.run_loop()
        self.predict.assert_not_called()
        self.policy.reset.assert_not_called()


if __name__=='__main__': unittest.main()
