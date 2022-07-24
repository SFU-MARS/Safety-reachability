def carmodel(self, simulator, i, dt):
    # self._position_nk2, self._speed_nk1,
    # self._acceleration_nk1, self._heading_nk1,
    # self._angular_speed_nk1, self._angular_acceleration_nk1

    self.acceleration_nk1 = i[0]
    self.angular_speed_nk1 = i[1]

    self.speed_nk1_next = simulator.start_config.trainable_variables[1] + self.acceleration_nk1 * dt
    self.heading_nk1_next = simulator.start_config.trainable_variables[2] + self.angular_speed_nk1 * dt

    # x0=np.array(start_config[0].read_value())[0,0][0]
    x0 = simulator.start_config.trainable_variables[0][0][0][0]
    # y0 = np.array(start_config[0].read_value())[0, 0][1]
    y0 = simulator.start_config.trainable_variables[0][0][0][1]
    position_nk1_next0 = []
    position_nk1_next1 = []
    # position_nk1_next0 = start_config[1].numpy()*np.cos(start_config[3].numpy())*dt+x0,
    position_nk1_next0 = simulator.start_config.trainable_variables[1] * np.cos(
        simulator.start_config.trainable_variables[3][0][0]) * dt + x0
    # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
    # position_nk1_next0=list(position_nk1_next0)
    position_nk1_next1 = simulator.start_config.trainable_variables[1] * np.sin(
        simulator.start_config.trainable_variables[3][0][0]) * dt + y0
    # position_nk1_next1 = start_config[1].numpy()*np.sin(start_config[3].numpy())*dt+y0,
    # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
    # position_nk1_next1 = list(position_nk1_next1)

    x0 = position_nk1_next0
    y0 = position_nk1_next1
    # position_nk1_next0 = []
    # position_nk1_next1 = []
    # position_nk1_next0 = start_config[1].numpy() * np.cos(start_config[3].numpy()) * dt + x0,
    # # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
    # position_nk1_next0 = list(position_nk1_next0)
    # position_nk1_next1 = start_config[1].numpy() * np.sin(start_config[3].numpy()) * dt + y0,
    # # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
    # position_nk1_next1 = list(position_nk1_next1)
    # start_config[0], start_config[1], start_config[2], start_config[3], start_config[4] = [position_nk1_next0,
    #                                                                                        position_nk1_next1], self.speed_nk1_next, self.acceleration_nk1, self.heading_nk1_next, self.angular_speed_nk1

    return position_nk1_next0, position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next
