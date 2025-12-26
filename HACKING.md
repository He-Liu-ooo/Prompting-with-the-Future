# HACKING

## Environment setup

- CUDA: tested with CUDA 11.8.
- `pytorch3d`: install via conda:

```bash
conda install pytorch3d::pytorch3d -c pytorch3d -c pytorch -c conda-forge
```

- SAM2: download the upstream README from https://github.com/facebookresearch/sam2 and place it under the `sam2/` folder before running:

```bash
pip install -e sam2
```

- COLMAP: install a usable binary (conda-forge has a package). Verify it runs:

```bash
conda install -c conda-forge colmap
colmap -h  # make sure runnable
```

- e3nn: required version for this repo is `0.5.1`:

```bash
pip install "e3nn==0.5.1"
```

- NumPy: if you run into incompatibilities, downgrade to `1.26.4`:

```bash
pip uninstall -y numpy
pip install "numpy==1.26.4"
```

## Run notes

- Avoid embedding raw images as base64 in prompts — this explodes token counts and is inefficient.
- The repo includes an alternative prompt helper: `utils/prompt_deepseek.py`. Use Deepseek's API if you want to avoid OpenAI image encoding; see https://platform.deepseek.com/usage for details.

## Performance bottlenecks

If you profile the code, the main slow paths we observed are related to the physics / controller pipeline. Key hotspots and files:

- Scene stepping (PhysX / ManiSkill):
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/envs/sapien_env.py` — environment stepping / wrapper
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/envs/scene.py` — `ManiSkillScene.step()` calls into PhysX
    ```
    class ManiSkillScene:
      def __init__(
          self,
          sub_scenes: Optional[list[sapien.Scene]] = None,
          sim_config: SimConfig = SimConfig(),
          debug_mode: bool = True,
          device: Device = None,
          parallel_in_single_scene: bool = False,
          backend: BackendInfo = None,
      ):    
          self.px: Union[physx.PhysxCpuSystem, physx.PhysxGpuSystem] = self.sub_scenes[
              0
          ].physx_system

      def step(self):
          self.px.step()
    ```

- Agent / controller overheads (sanity checks, kinematics):
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/agents/base_agent.py` — `BaseAgent.set_action()` validation
    ```
    class BaseAgent:
    def set_action(self, action):
        """
        Set the agent's action which is to be executed in the next environment timestep.
        This is essentially a wrapper around the controller's set_action method.
        """
        if not self.scene.gpu_sim_enabled:
            if np.isnan(action).any():
                raise ValueError("Action cannot be NaN. Environment received:", action)
        self.controller.set_action(action)
    ```
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/agents/controllers/base_controller.py` — `CombinedController.set_action()` does shape checks and splits actions
    ```
    class CombinedController(DictController):
    def set_action(self, action: np.ndarray):
        # Sanity check
        # TODO (stao): optimization, do we really need this sanity check? Does gymnasium already do this for us
        print("here!")
        if self.scene.num_envs > 1:
            action_dim = self.action_space.shape[1]
        else:
            action_dim = self.action_space.shape[0]
        assert action.shape == (
            self.scene.num_envs,
            action_dim,
        ), f"Received action of shape {action.shape} but expected shape ({self.scene.num_envs}, {action_dim})"
        for uid, controller in self.controllers.items():
            start, end = self.action_mapping[uid]
            controller.set_action(action[:, start:end])
    ```
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/agents/controllers/pd_joint_pos.py`
    ```
    class PDJointPosMimicController(PDJointPosController)
    ```
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/agents/controllers/pd_ee_pose.py`
    ```
    class PDEEPosController(PDJointPosController)
    ```
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/agents/controllers/passive_controller.py`
    ```
    class PassiveController(BaseController)
    ```

- Kinematics helpers (expensive Jacobian/IK work):
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/mani_skill/agents/controllers/utils/kinematics.py`
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/pytorch_kinematics/chain.py`
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/pytorch_kinematics/jacobian.py`
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/pytorch_kinematics/frame.py`
  - `/home/hel19/workspace/env/miniconda3/envs/pwf/lib/python3.9/site-packages/pytorch_kinematics/transforms/rotation_conversions.py`
    
