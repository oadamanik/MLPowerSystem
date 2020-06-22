from tkinter import *
import pandapower as pp
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from timeseriessim import TimeSeries
from mlearninglib import KMeans, KNN, accuracy
from testnetwork import test_network


class MainPage:

    def __init__(self, master):
        self.master = master

        # Number of simulation time steps
        self.n_time_steps = 10

        # Initialize simulation
        self.time_series_simulation = TimeSeries(test_network(), self.n_time_steps)

        # Initialize number of operating cases, number of buses, k for k-means, and k for kNN
        self.n_cases = 0
        self.n_bus = 9
        self.k_kmeans = 6
        self.k_knn = 3

        # Establish canvas
        self.canvas = Canvas(self.master, height=500, width=500)
        self.canvas.pack()

        # Establish frame of the main page
        self.frame = Frame(self.master)
        self.frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.8)

        # Time step widgets
        self.n_time_steps_label = Label(self.frame, text='Type number of simulation steps',
                                        font=("Arial", "10"))
        self.n_time_steps_label.place(relheight=0.05, relwidth=0.75, relx=0.0, rely=0.05)
        
        self.n_time_steps_entry = Entry(self.frame, font=("Arial", "10"), relief='sunken')
        self.n_time_steps_entry.insert(END, str(self.n_time_steps))
        self.n_time_steps_entry.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.05)
        
        # Time series simulation widgets
        self.run_time_series_button = Button(self.frame, text="Run time series simulation",
                                             command=self.run_time_series)
        self.run_time_series_button.place(relheight=0.1, relwidth=0.75, relx=0.0, rely=0.10)
        
        self.run_time_series_label = Label(self.frame, text=None, font=("Arial", "10", 'italic'),
                                           fg='white', relief='sunken')
        self.run_time_series_label.place(relheight=0.1, relwidth=0.25, relx=0.75, rely=0.10)

        self.about_frame = Frame(self.master)
        self.about_frame.place(relx=0.0, rely=0.75, relwidth=1, relheight=0.2)

        code_about = "KTH Royal Institute of Technology\n" \
                "EH2745 Computer Applications in Power Systems\n" \
                  "Assignment 2\n" \
                  "Created by oadamanik\n" \
                  "Version 1.0\n"

        self.code_about = Label(self.about_frame, text=code_about,
                                font=("Arial", "10"), justify="center")
        self.code_about.place(relheight=1, relwidth=1, relx=0.0, rely=0.0)

        self.exit_button = Button(self.frame, text='EXIT',
                                font=("Arial", "10"), command=self.canvas.quit)
        self.exit_button.place(relheight=0.05, relwidth=1, relx=0.0, rely=0.8)

    def run_time_series(self):
        self.n_time_steps = int(self.n_time_steps_entry.get())
        self.time_series_simulation.n_time_steps = self.n_time_steps
        print('Time steps: ', self.n_time_steps)
        
        # Create the main output dictionary
        output_dir = os.path.join(tempfile.gettempdir(), "time_series")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Create the output dictionary for the base configuration, high load simulation result
        output_dir_base_high = os.path.join(output_dir, "base_high")
        print("Results can be found in your local temp folder: {}".format(output_dir_base_high))
        if not os.path.exists(output_dir_base_high):
            os.mkdir(output_dir_base_high)

        # Running the time series simulation for the base configuration, high load
        self.time_series_simulation.time_series(output_dir_base_high, 'high')
        self.n_cases += 1

        # Create the output dictionary for the base configuration, low load simulation result
        output_dir_base_low = os.path.join(output_dir, "base_low")
        print("Results can be found in your local temp folder: {}".format(output_dir_base_low))
        if not os.path.exists(output_dir_base_low):
            os.mkdir(output_dir_base_low)

        # Running the time series simulation for the base configuration, low load
        # Re-initialize the pandapower network
        self.time_series_simulation.net = test_network()
        self.time_series_simulation.time_series(output_dir_base_low, 'low')
        self.n_cases += 1

        # Create the output dictionary for the Gen 3 disconnected configuration, high load simulation result
        output_dir_gen_disc_high = os.path.join(output_dir, "gen_disc_high")
        print("Results can be found in your local temp folder: {}".format(output_dir_gen_disc_high))
        if not os.path.exists(output_dir_gen_disc_high):
            os.mkdir(output_dir_gen_disc_high)

        # Run the time series simulation for the Gen 3 disconnected configuration, high load simulation result
        # Re-initialize the pandapower network
        self.time_series_simulation.net = test_network()
        gen_3_idx = pp.get_element_index(self.time_series_simulation.net, 'sgen', 'Gen 3')
        # Put Gen 3 out of service
        self.time_series_simulation.net.sgen.in_service[gen_3_idx] = False
        self.time_series_simulation.time_series(output_dir_gen_disc_high, 'high')
        # Put Gen 3 back in service
        self.time_series_simulation.net.sgen.in_service[gen_3_idx] = True
        self.n_cases += 1

        # Create the output dictionary for the Gen 3 disconnected configuration, low load simulation result
        output_dir_gen_disc_low = os.path.join(output_dir, "gen_disc_low")
        print("Results can be found in your local temp folder: {}".format(output_dir_gen_disc_low))
        if not os.path.exists(output_dir_gen_disc_low):
            os.mkdir(output_dir_gen_disc_low)

        # Run the time series simulation for the Gen 3 disconnected configuration, low load simulation result
        # Re-initialize the pandapower network
        self.time_series_simulation.net = test_network()
        # Put Gen 3 out of service
        self.time_series_simulation.net.sgen.in_service[1] = False
        self.time_series_simulation.time_series(output_dir_gen_disc_low, 'low')
        # Put Gen 3 back to service
        self.time_series_simulation.net.sgen.in_service[1] = True
        self.n_cases += 1

        # Create the output dictionary for the Line 5-6 disconnected configuration, high load simulation result
        output_dir_line_disc_high = os.path.join(tempfile.gettempdir(), "time_series", "line_disc_high")
        print("Results can be found in your local temp folder: {}".format(output_dir_line_disc_high))
        if not os.path.exists(output_dir_line_disc_high):
            os.mkdir(output_dir_line_disc_high)

        # Run the time series simulation for the Line 5-6 disconnected configuration, high load simulation result
        # Re-initialize the pandapower network
        self.time_series_simulation.net = test_network()
        line_5_6_idx = pp.get_element_index(self.time_series_simulation.net, 'line', 'Line 5-6')
        # Put Line 5-6 out of service
        self.time_series_simulation.net.line.in_service[line_5_6_idx] = False
        self.time_series_simulation.time_series(output_dir_line_disc_high, 'high')
        # Put Line 5-6 back to service
        self.time_series_simulation.net.line.in_service[line_5_6_idx] = True
        self.n_cases += 1

        # Create the output dictionary for the Line 5-6 disconnected configuration, low load simulation result
        output_dir_line_disc_low = os.path.join(tempfile.gettempdir(), "time_series", "line_disc_low")
        print("Results can be found in your local temp folder: {}".format(output_dir_line_disc_low))
        if not os.path.exists(output_dir_line_disc_low):
            os.mkdir(output_dir_line_disc_low)

        # Run the time series simulation for the Line 5-6 disconnected configuration,low load simulation result
        # Re-initialize the pandapower network
        self.time_series_simulation.net = test_network()
        line_5_6_idx = pp.get_element_index(self.time_series_simulation.net, 'line', 'Line 5-6')
        # Put Line 5-6 out of service
        self.time_series_simulation.net.line.in_service[line_5_6_idx] = False
        self.time_series_simulation.time_series(output_dir_line_disc_low, 'low')
        # Put Line 5-6 back to service
        self.time_series_simulation.net.line.in_service[line_5_6_idx] = True
        self.n_cases += 1

        # Create data frame for high load
        voltage_base_high_file = os.path.join(output_dir_base_high, "res_bus", "vm_pu.xls")
        self.voltage_base_high_df = pd.read_excel(voltage_base_high_file, index_col=0)

        angle_base_high_file = os.path.join(output_dir_base_high, "res_bus", "va_degree.xls")
        self.angle_base_high_df = pd.read_excel(angle_base_high_file, index_col=0)

        self.base_high_df = pd.concat([self.voltage_base_high_df,
                                       self.angle_base_high_df],
                                      axis=1, ignore_index=True)
        self.base_high_df['os_label'] = 0

        # Create data frame for low load
        voltage_base_low_file = os.path.join(output_dir_base_low, "res_bus", "vm_pu.xls")
        self.voltage_base_low_df = pd.read_excel(voltage_base_low_file, index_col=0)

        angle_base_low_file = os.path.join(output_dir_base_low, "res_bus", "va_degree.xls")
        self.angle_base_low_df = pd.read_excel(angle_base_low_file, index_col=0)

        self.base_low_df = pd.concat([self.voltage_base_low_df,
                                      self.angle_base_low_df],
                                     axis=1, ignore_index=True)
        self.base_low_df['os_label'] = 1

        # Create data frame for high load, Gen 3 disconnected
        voltage_gen_disc_high_file = os.path.join(output_dir_gen_disc_high, "res_bus", "vm_pu.xls")
        self.voltage_gen_disc_high_df = pd.read_excel(voltage_gen_disc_high_file, index_col=0)

        angle_gen_disc_high_file = os.path.join(output_dir_gen_disc_low, "res_bus", "va_degree.xls")
        self.angle_gen_disc_high_df = pd.read_excel(angle_gen_disc_high_file, index_col=0)

        self.gen_disc_high_df = pd.concat([self.voltage_gen_disc_high_df,
                                           self.angle_gen_disc_high_df],
                                          axis=1, ignore_index=True)
        self.gen_disc_high_df['os_label'] = 2

        # Create data frame for low load, Gen 3 disconnected
        voltage_gen_disc_low_file = os.path.join(output_dir_gen_disc_low, "res_bus", "vm_pu.xls")
        self.voltage_gen_disc_low_df = pd.read_excel(voltage_gen_disc_low_file, index_col=0)

        angle_gen_disc_low_file = os.path.join(output_dir_gen_disc_low, "res_bus", "va_degree.xls")
        self.angle_gen_disc_low_df = pd.read_excel(angle_gen_disc_low_file, index_col=0)

        self.gen_disc_low_df = pd.concat([self.voltage_gen_disc_low_df,
                                          self.angle_gen_disc_low_df],
                                         axis=1, ignore_index=True)
        self.gen_disc_low_df['os_label'] = 3

        # Create data frame for high load, Line 5-6 disconnected
        voltage_line_disc_high_file = os.path.join(output_dir_line_disc_high, "res_bus", "vm_pu.xls")
        self.voltage_line_disc_high_df = pd.read_excel(voltage_line_disc_high_file, index_col=0)

        angle_line_disc_high_file = os.path.join(output_dir_line_disc_low, "res_bus", "va_degree.xls")
        self.angle_line_disc_high_df = pd.read_excel(angle_line_disc_high_file, index_col=0)

        self.line_disc_high_df = pd.concat([self.voltage_line_disc_high_df,
                                            self.angle_line_disc_high_df],
                                           axis=1, ignore_index=True)
        self.line_disc_high_df['os_label'] = 4

        # Create data frame for low load, Line 5-6 disconnected
        voltage_line_disc_low_file = os.path.join(output_dir_line_disc_low, "res_bus", "vm_pu.xls")
        self.voltage_line_disc_low_df = pd.read_excel(voltage_line_disc_low_file, index_col=0)

        angle_line_disc_low_file = os.path.join(output_dir_line_disc_low, "res_bus", "va_degree.xls")
        self.angle_line_disc_low_df = pd.read_excel(angle_line_disc_low_file, index_col=0)

        self.line_disc_low_df = pd.concat([self.voltage_line_disc_low_df,
                                           self.angle_line_disc_low_df],
                                          axis=1, ignore_index=True)
        self.line_disc_low_df['os_label'] = 5

        # Combine dataset
        self.dataset_df_with_label = pd.concat([self.base_high_df, self.base_low_df,
                                                self.gen_disc_high_df, self.gen_disc_low_df,
                                                self.line_disc_high_df, self.line_disc_low_df],
                                               axis=0, ignore_index=True)

        # Dataset without label
        self.dataset_df = self.dataset_df_with_label.drop(['os_label'], axis=1)

        # Initialize the normalization
        dataset_df_for_normalization = self.dataset_df

        self.dataset_normalized_df = self.dataset_df.copy()

        # Normalization, excluding the slack bus voltage and angle
        for i in range(1, self.n_bus):
            self.dataset_normalized_df[i] = np.divide(
                dataset_df_for_normalization[i] - dataset_df_for_normalization[i].min(),
                dataset_df_for_normalization[i].max() - dataset_df_for_normalization[i].min())

        for i in range(self.n_bus + 1, 2 * self.n_bus):
            self.dataset_normalized_df[i] = np.divide(
                dataset_df_for_normalization[i] - dataset_df_for_normalization[i].min(),
                dataset_df_for_normalization[i].max() - dataset_df_for_normalization[i].min())

        self.dataset_normalized_df_with_label = self.dataset_normalized_df.copy()
        self.dataset_normalized_df_with_label['os_label'] = self.dataset_df_with_label['os_label'].copy()

        # Fraction of the training set
        coeff_train = 0.8
        n_train = int(coeff_train * self.n_time_steps)

        # Normalized training set
        self.train_dataset_normalized_df_with_label = pd.concat(
            [self.dataset_normalized_df_with_label[:n_train],
             self.dataset_normalized_df_with_label[self.n_time_steps:n_train + self.n_time_steps],
             self.dataset_normalized_df_with_label[2 * self.n_time_steps:n_train + 2 * self.n_time_steps],
             self.dataset_normalized_df_with_label[3 * self.n_time_steps:n_train + 3 * self.n_time_steps],
             self.dataset_normalized_df_with_label[4 * self.n_time_steps:n_train + 4 * self.n_time_steps],
             self.dataset_normalized_df_with_label[5 * self.n_time_steps:n_train + 5 * self.n_time_steps]],
            axis=0, ignore_index=True)

        # Normalized test set
        self.test_dataset_normalized_df_with_label = pd.concat(
            [self.dataset_normalized_df_with_label[n_train:self.n_time_steps],
             self.dataset_normalized_df_with_label[n_train + self.n_time_steps:2 * self.n_time_steps],
             self.dataset_normalized_df_with_label[n_train + 2 * self.n_time_steps:3 * self.n_time_steps],
             self.dataset_normalized_df_with_label[n_train + 3 * self.n_time_steps:4 * self.n_time_steps],
             self.dataset_normalized_df_with_label[n_train + 4 * self.n_time_steps:5 * self.n_time_steps],
             self.dataset_normalized_df_with_label[n_train + 5 * self.n_time_steps:6 * self.n_time_steps]],
            axis=0, ignore_index=True)

        # Shuffling the normalized dataset
        # self.dataset_normalized_df_with_label = self.dataset_normalized_df_with_label.sample(
        #     frac=1).reset_index(drop=True)
        # self.train_dataset_normalized_df_with_label = self.train_dataset_normalized_df_with_label.sample(
        #     frac=1).reset_index(drop=True)
        # self.test_dataset_normalized_df_with_label = self.test_dataset_normalized_df_with_label.sample(
        #     frac=1).reset_index(drop=True)

        # Export dataframe
        dataset_df_filename = "dataset_df.xlsx"
        dataset_normalized_df_filename = "dataset_normalized.xlsx"
        train_dataset_normalized_df_filename = "train_dataset_normalized.xlsx"
        test_dataset_normalized_df_filename = "test_dataset_normalized.xlsx"
        self.dataset_df_with_label.to_excel(dataset_df_filename)
        self.dataset_normalized_df_with_label.to_excel(dataset_normalized_df_filename)
        self.train_dataset_normalized_df_with_label.to_excel(train_dataset_normalized_df_filename)
        self.test_dataset_normalized_df_with_label.to_excel(test_dataset_normalized_df_filename)
        print(dataset_df_filename, 'has been created in the project directory')
        print(dataset_normalized_df_filename, 'has been created in the project directory')
        print(train_dataset_normalized_df_filename, 'has been created in the project directory')
        print(test_dataset_normalized_df_filename, 'has been created in the project directory')

        # Time series simulation widgets
        self.run_time_series_label = Label(self.frame, text='Success', font=("Arial", "10", 'italic'),
                                           bg='green', fg='white', relief='sunken')
        self.run_time_series_label.place(relheight=0.1, relwidth=0.25, relx=0.75, rely=0.10)

        # Plot widgets
        self.plot_simulation_button = Button(self.frame, text="Plot simulation result",
                                             command=self.plot_simulation_result)
        self.plot_simulation_button.place(relheight=0.05, relwidth=0.75, relx=0.0, rely=0.20)

        self.plot_simulation_label = Label(self.frame, text=None, font=("Arial", "10", 'italic'),
                                           fg='white', relief='sunken')
        self.plot_simulation_label.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.20)

        # k-means widgets
        self.k_means_title_label = Label(self.frame, text='k-means clustering',
                                        font=("Arial", "10"))
        self.k_means_title_label.place(relheight=0.05, relwidth=1, relx=0.0, rely=0.35)

        self.k_means_label = Label(self.frame, text='Insert the number of clusters k',
                                         font=("Arial", "10"))
        self.k_means_label.place(relheight=0.05, relwidth=0.75, relx=0.0, rely=0.40)

        self.k_means_entry = Entry(self.frame, font=("Arial", "10"), relief='sunken')
        self.k_means_entry.insert(END, str(self.k_kmeans))
        self.k_means_entry.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.40)

        self.k_means_button = Button(self.frame, text="Run k-means algorithm",
                                             command=self.k_means_clustering)
        self.k_means_button.place(relheight=0.05, relwidth=0.75, relx=0.0, rely=0.45)

        self.k_means_result_label = Label(self.frame, text=None, font=("Arial", "10", 'italic'),
                                           fg='white', relief='sunken')
        self.k_means_result_label.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.45)

        # k-NN widgets
        self.k_nn_title_label = Label(self.frame, text='k-NN classification',
                                         font=("Arial", "10"))
        self.k_nn_title_label.place(relheight=0.05, relwidth=1, relx=0.0, rely=0.55)

        self.k_nn_label = Label(self.frame, text='Insert the number of k nearest neighbors',
                                   font=("Arial", "10"))
        self.k_nn_label.place(relheight=0.05, relwidth=0.75, relx=0.0, rely=0.60)

        self.k_nn_entry = Entry(self.frame, font=("Arial", "10"), relief='sunken')
        self.k_nn_entry.insert(END, str(self.k_knn))
        self.k_nn_entry.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.60)

        self.k_nn_button = Button(self.frame, text="Run k-NN algorithm",
                                     command=self.knn)
        self.k_nn_button.place(relheight=0.05, relwidth=0.75, relx=0.0, rely=0.65)

        self.k_nn_result_label = Label(self.frame, text=None, font=("Arial", "10", 'italic'),
                                          fg='white', relief='sunken')
        self.k_nn_result_label.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.65)

    def plot_simulation_result(self):

        self.fig, ax = plt.subplots(nrows=6, figsize=(6, 12))

        # Plotting
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

        voltage_df = [self.voltage_base_high_df, self.voltage_base_low_df,
                      self.voltage_gen_disc_high_df, self.voltage_gen_disc_low_df,
                      self.voltage_line_disc_high_df, self.voltage_line_disc_low_df]

        angle_df = [self.angle_base_high_df, self.angle_base_low_df,
                    self.angle_gen_disc_high_df, self.angle_gen_disc_low_df,
                    self.angle_line_disc_high_df, self.angle_line_disc_low_df]

        title_list = ['Base configuration, high load',
                      'Base configuration, low load',
                      'Gen 3 disconnected, high load',
                      'Gen 3 disconnected, low load',
                      'Line 5-6 disconnected, high load',
                      'Line 5-6 disconnected, low load']

        for j in range(0, self.n_cases):
            for i in range(0, self.n_bus):
                ax[j].scatter(voltage_df[j][i], angle_df[j][i], c=color[i], s=5, label='Bus {}'.format(i + 1))
                box = ax[j].get_position()
                ax[j].set_position([-0.075, box.y0, box.width, box.height])
                ax[j].set_title(title_list[j])
                ax[j].set_xlabel('Voltage (p.u.)')
                ax[j].set_ylabel('Angle (deg.)')

        handles, labels = ax[0].get_legend_handles_labels()
        self.fig.legend(handles, labels, loc='center left',
                   ncol=1, fancybox=True, shadow=True)
        plt.show()

        fig_file_name = 'plot.png'
        self.fig.savefig(fig_file_name)

        self.plot_simulation_label = Label(self.frame, text='Success', font=("Arial", "10", 'italic'),
                                           bg='green', fg='white', relief='sunken')
        self.plot_simulation_label.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.20)

        self.plot_simulation_label = Label(self.frame, text='{} has been created'.format(fig_file_name),
                                           font=("Arial", "10", 'italic'), relief='groove')
        self.plot_simulation_label.place(relheight=0.05, relwidth=1, relx=0.0, rely=0.25)

    def k_means_clustering(self):
        # k-means clustering
        self.k_kmeans = int(self.k_means_entry.get())
        print('k for k-means: ', self.k_kmeans)

        self.x_set = self.dataset_normalized_df_with_label.drop(['os_label'], axis=1).to_numpy()
        self.clusterization = KMeans(k=self.k_kmeans, max_iteration=300)
        y_pred = self.clusterization.predict(self.x_set)
        print('Clustering result: \n', self.clusterization.clusters)

        self.k_means_result_label = Label(self.frame, text='Success', font=("Arial", "10", 'italic'),
                                          bg='green', fg='white', relief='sunken')
        self.k_means_result_label.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.45)

    def knn(self):
        self.k_knn = int(self.k_nn_entry.get())
        print('k for kNN classification: ', self.k_knn)

        # Create the training set for normalized
        x_train = self.train_dataset_normalized_df_with_label.drop(['os_label'], axis=1).to_numpy()
        y_train = self.train_dataset_normalized_df_with_label['os_label'].to_numpy()
        # Create the test set for normalized
        x_test = self.test_dataset_normalized_df_with_label.drop(['os_label'], axis=1).to_numpy()
        y_test = self.test_dataset_normalized_df_with_label['os_label'].to_numpy()

        classification = KNN(k=self.k_knn)
        classification.fit(x_train, y_train)
        prediction = classification.predict(x_test)
        print('y_test', y_test)
        print('Test result: ', prediction)
        print("Accuracy: {}%".format(accuracy(y_test, prediction)*100))

        self.k_nn_result_label = Label(self.frame, text='Success', font=("Arial", "10", 'italic'),
                                          bg='green', fg='white', relief='sunken')
        self.k_nn_result_label.place(relheight=0.05, relwidth=0.25, relx=0.75, rely=0.65)
