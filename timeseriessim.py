import pandas as pd
import numpy as np
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl


class TimeSeries:

    def __init__(self, net, n_time_steps):
        # Initialize the power network and the time steps
        self.net = net
        self.n_time_steps = n_time_steps

    def create_data_source(self, mode):
        # The function to create data frame for the loads
        profiles = pd.DataFrame()

        # For high load, 5-10% more than the initial value
        if mode == 'high':
            for i in range(len(self.net.load)):
                profiles['load_{}_p'.format(str(i))] = (1.05 + (0.05 * np.random.random(
                    self.n_time_steps))) * self.net.load.p_mw[i]
                profiles['load_{}_q'.format(str(i))] = (1.05 + (0.05 * np.random.random(
                    self.n_time_steps))) * self.net.load.q_mvar[i]

        # For low load, 5-10% less than the initial value
        elif mode == 'low':
            for i in range(len(self.net.load)):
                profiles['load_{}_p'.format(str(i))] = (0.90 + (0.05 * np.random.random(
                    self.n_time_steps))) * self.net.load.p_mw[i]
                profiles['load_{}_q'.format(str(i))] = (0.90 + (0.05 * np.random.random(
                    self.n_time_steps))) * self.net.load.q_mvar[i]

        ds = DFData(profiles)
        return profiles, ds

    def create_controllers(self, ds):
        # Change the load in the pandapower network based on the provided data source
        for i in range(len(self.net.load)):
            ConstControl(self.net, element='load', variable='p_mw', element_index=[i],
                         data_source=ds, profile_name=['load_{}_p'.format(str(i))])
            ConstControl(self.net, element='load', variable='q_mvar', element_index=[i],
                         data_source=ds, profile_name=['load_{}_q'.format(str(i))])

        return self.net

    def create_output_writer(self, output_dir):
        ow = OutputWriter(self.net, self.n_time_steps, output_path=output_dir,
                          output_file_type=".xls", log_variables=list())
        # Logging information of the bus voltages and angles
        ow.log_variable('res_bus', 'vm_pu')
        ow.log_variable('res_bus', 'va_degree')

        return ow

    def time_series(self, output_dir, mode):
        profiles, ds = self.create_data_source(mode)
        self.create_controllers(ds)
        time_steps = range(0, self.n_time_steps)
        self.create_output_writer(output_dir)
        run_timeseries(self.net, time_steps, calculate_voltage_angles=True)