"""
Parameters used in the training process of the temporal graph foundation model.
"""

# supported_properties = {"edge_gs": "Edge_GS", "density_gs": "Density_GS", "node_reg": "Node_REG"}
supported_properties = {"node_gs": "Node_GS"}
# supported_properties = {"new_node_reg_normilized_ts": "New_Nodes_REG_Normalized_TS",
#                        "node_deg_5_reg_normilized_ts": "Node_Deg_5_REG_Normalized_TS",
# "edge_count_reg_normalized_ts": "Edge_Count_REG_Normalized_TS"}

# supported_properties = {"edge_count_reg_normalized_ts": "Edge_Count_REG_Normalized_TS"}

# supported_properties = {"edge_gs": "Edge_GS", "node_gs": "Node_GS", "lconcomp_gs": "LargeConComp_GS"}
# BC -> Binary Classification
# REG -> Regression
### Always use BC tasks first and then add REG at the end ###

# supported_properties_type = ["BC", "BC", "BC"]
supported_properties_type = ["BC"]
# supported_properties_type = ["REG", "REG", "REG"]
# supported_properties_type = ["REG"]
# supported_properties_type = ["BC", "BC", "BC", "BC", "BC"]

# Multi Loss combination methods
MULTI_LOSS_WEIGHTED_SUM = "weighted_sum"
MULTI_LOSS_DYNAMIC_BALANCING = "dynamic_balancing"
MULTI_LOSS_UNCERTAINTY_BASED = "uncertainty_based"
MULTI_LOSS_SIMPLE_SUM = "simple_sum"
