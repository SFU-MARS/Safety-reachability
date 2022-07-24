def reset(self, seed=-1):
    """Reset the simulator. Optionally takes a seed to reset
    the simulator's random state."""
    if seed != -1:
        self.rng.seed(seed)

    # Note: Obstacle map must be reset independently of the fmm map.
    # Sampling start and goal may depend on the updated state of the
    # obstacle map. Updating the fmm map depends on the newly sampled goal.
    reset_start = True
    while reset_start:
        self._reset_obstacle_map(self.rng)  # Do nothing here

        self._reset_start_configuration(self.rng)  # Reset self.start_config
        # Reset self.goal_config. If there is no available goals, reset_start = True, then reset the start again.
        reset_start = self._reset_goal_configuration(self.rng)

        # Manually restart the start and goal (only work for single goal)
        # reset_start = self._reset_start_goal_manually(start_pos=[8.65, 50.25], goal_pos=[8.60, 47.15])

    self._update_fmm_map()  # Compute fmm_angle and fmm_goal, wrap it into voxel func

    # Initiate and update a reachability map (reach_avoid or avoid)
    if self.params.cost == 'reachability':
        self._get_reachability_map()

    # Update objective functions, may include reachability cost
    self._update_obj_fn()

    self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
    self.obj_val = np.inf
    self.vehicle_data = {}