import copy
import importlib
from multiprocessing import Array, Value, Lock
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.hand_joint_limits import resolve_hand_joint_limits
from unitree_lerobot.utils.constants import (G1_RIGHT_ARM_2CAM_CONFIG, G1_DEX1_CONFIG,
    G1_DEX3_CONFIG, G1_BRAINCO_CONFIG, G1_INSPIRE_CONFIG)


class NormalizationRegressionTests(unittest.TestCase):
    def test_real_processor_creation_and_checkpoint_roundtrip(self):
        for names, hand, special in [
            (G1_RIGHT_ARM_2CAM_CONFIG.motors, '', False),
            (G1_DEX1_CONFIG.motors, 'dex1', False),
            (G1_INSPIRE_CONFIG.motors, 'inspire1', False),
            (G1_DEX3_CONFIG.motors[7:14]+G1_DEX3_CONFIG.motors[21:], 'dex3', True),
            (G1_BRAINCO_CONFIG.motors[7:14]+G1_BRAINCO_CONFIG.motors[20:], 'brainco', True),
            (G1_DEX3_CONFIG.motors, 'dex3', True),
        ]:
            with self.subTest(hand=hand, dimension=len(names)):
                n=len(names)
                meta=SimpleNamespace(features={'observation.state':{'names':[names],'shape':[n]}})
                cfg=ACTConfig(device='cpu',input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(n,))},
                              output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(n,))})
                stats={k:{'mean':torch.zeros(n),'std':torch.ones(n)} for k in ('observation.state','action')}
                pre,post=make_pre_post_processors(cfg,dataset_meta=meta,dataset_stats=stats,ee=hand)
                sample={'observation.state':torch.full((1,n),0.2)}
                expected=pre(copy.deepcopy(sample))['observation.state']
                self.assertEqual(resolve_hand_joint_limits(meta.features) is not None,special)
                with tempfile.TemporaryDirectory() as tmp:
                    pre.save_pretrained(tmp);post.save_pretrained(tmp)
                    loaded,_=make_pre_post_processors(cfg,pretrained_path=tmp,dataset_meta=meta,ee=hand)
                    torch.testing.assert_close(loaded(copy.deepcopy(sample))['observation.state'],expected)
                    if special:
                        cfg.hand_joint_limit_normalization=False
                        with self.assertRaises(ValueError):
                            make_pre_post_processors(cfg,pretrained_path=tmp,dataset_meta=meta,ee=hand)

    def test_disabled_legacy_checkpoint_remains_disabled(self):
        names=G1_DEX3_CONFIG.motors[7:14]+G1_DEX3_CONFIG.motors[21:];n=len(names)
        meta=SimpleNamespace(features={'observation.state':{'names':[names],'shape':[n]}})
        cfg=ACTConfig(device='cpu',hand_joint_limit_normalization=False,
                      input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(n,))},
                      output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(n,))})
        stats={k:{'mean':torch.zeros(n),'std':torch.ones(n)} for k in ('observation.state','action')}
        pre,post=make_pre_post_processors(cfg,dataset_meta=meta,dataset_stats=stats,ee='dex3')
        with tempfile.TemporaryDirectory() as tmp:
            pre.save_pretrained(tmp);post.save_pretrained(tmp)
            make_pre_post_processors(cfg,pretrained_path=tmp,dataset_meta=meta,ee='dex3')
            cfg.hand_joint_limit_normalization=True
            with self.assertRaises(ValueError):
                make_pre_post_processors(cfg,pretrained_path=tmp,dataset_meta=meta,ee='dex3')


class ControllerStartupTests(unittest.TestCase):
    def test_actual_constructors_seed_commands_before_start(self):
        specs=[('robot_hand_unitree','Dex3_1_Controller',7),('robot_hand_unitree','Dex1_1_Gripper_Controller',1),
               ('robot_hand_brainco','Brainco_Controller',6),('robot_hand_inspire','Inspire_Controller',6)]
        for filename,clsname,dof in specs:
            for value in (0.0,0.35):
                with self.subTest(controller=clsname,value=value):
                    module=importlib.import_module('unitree_lerobot.eval_robot.robot_control.'+filename)
                    commands=[Value('d',-1) if dof==1 else Array('d',[-1]*dof) for _ in range(2)]
                    state=Array('d',2*dof);action=Array('d',2*dof)
                    msg=SimpleNamespace(states=[SimpleNamespace(q=value) for _ in range(12)],
                                        motor_state=[SimpleNamespace(q=value) for _ in range(7)])
                    starts=[]
                    class Worker:
                        def __init__(self,target,args=()): self.target,self.args=target,args
                        def start(self):
                            if self.target.__name__.startswith('_subscribe'):
                                try:self.target(*self.args)
                                except StopIteration:pass
                            else:
                                current=[c.value for c in commands] if dof==1 else list(commands[0][:])+list(commands[1][:])
                                starts.append(current)
                                self_outer.assertEqual(current,[value]*(2*dof))
                                self_outer.assertEqual(list(state[:]),[value]*(2*dof))
                                self_outer.assertEqual(list(action[:]),[value]*(2*dof))
                    self_outer=self
                    with patch.object(module,'ChannelFactoryInitialize'),patch.object(module,'ChannelPublisher'), \
                         patch.object(module,'ChannelSubscriber',return_value=Mock(Read=lambda:msg)), \
                         patch.object(module,'threading',SimpleNamespace(Thread=Worker, Event=threading.Event)),patch.object(module,'Process',Worker,create=True), \
                         patch.object(module.time,'sleep',side_effect=StopIteration):
                        getattr(module,clsname)(*commands,Lock(),state,action)
                    self.assertEqual(len(starts),1)

    def test_missing_state_does_not_start_worker(self):
        module=importlib.import_module('unitree_lerobot.eval_robot.robot_control.robot_hand_unitree')
        with patch.object(module,'ChannelFactoryInitialize'),patch.object(module,'ChannelPublisher') as publisher, \
             patch.object(module,'ChannelSubscriber'), \
             patch.object(module,'threading',SimpleNamespace(Thread=Mock(), Event=lambda:Mock(wait=Mock(return_value=False)))), \
             patch.object(module,'Process') as worker:
            with self.assertRaises(TimeoutError):
                module.Dex3_1_Controller(Array('d',7),Array('d',7))
            worker.assert_not_called()
            publisher.return_value.Write.assert_not_called()

if __name__=='__main__':unittest.main()
