Changelog
---------

0.1.0 (2025-07-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial port of the OmniIsaacGymEnvs ``DofbotReacher`` task to the Isaac Lab
  Direct RL workflow. The Dofbot 5-DoF arm is spawned directly from its
  bundled URDF description (no Nucleus-hosted USD asset required). Reward,
  observation and reset logic are ported from the original task.
