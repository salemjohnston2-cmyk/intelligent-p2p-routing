# intelligent-p2p-routing

Intelligent Peer-to-Peer Routing Framework
B.Sc. Computer Science Final Year Project
University of Abuja, Department of Computer Science (2026)
Author: Ayegba Salem Uredo (22/208CSC/263)

Overview
This repository contains the experimental implementation for an intelligent Peer-to-Peer routing framework designed to optimize Data Dissemination in dynamic network environments. Unlike conventional routing protocols that rely solely on hop count, this framework utilizes a Multilayer Perceptron (MLP) deep learning model to predict route quality based on five critical network metrics: latency, bandwidth, congestion, hop count, and packet loss.

Key Features
Dynamic Network Simulation: Generates realistic, fluctuating network topologies using NetworkX.
Deep Learning Route Selection: Trains a 5-32-16-1 Neural Network to approximate a multi-factor routing objective.
Comparative Analysis: Benchmarks intelligent routing aganst standard Shortest-Hop (Dijkstra) and Oracle (Upper Bound) strategies.
Performance Metrics: Evaluates Throughput, Packet Delivery Ratio (PDR), End-to-End Delay, and a custom Routing Efficiency metric.

Prerequisites
Python 3.8+
NumPy
NetworkX
Scikit-learn
Matplotlib

Installation & Usage
1.
Clone the repository:
git clone https://github.com/salemjohnston2-cmyk/intelligent-p2p-routing.git
2.
Install dependencies:
pip install -r http://requirements.txt
3.
Run the experiment:
python http://main.py

The script will generate the training dataset, train the model, run the simulation on 300 test pairs, and save the results to experiment_results.json and the figures/ directory.

Experimental Results
Model Accuracy: The MLP achieved an R² score of 0.9986 on unseen test scenarios.
Efficiency Improvement: Intelligent routing improved overall routing efficiency by 72.82% compared to conventional shortest-hop methods.
Throughput: Achieved a 24.88% increase in average throughput by intelligently bypassing congested links.

Citation
If you use this code for academic purposes, please cite:
Ayegba, S. U. (2026). An Intelligent Peer-to-Peer Framework for Efficient Data Dissemination Using Deep Learning-Based Routing. Department of Computer Science, University of Abuja.
