
�hroot"_tf_keras_network*�g{"name": "model_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_34"}, "name": "input_34", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden2", "inbound_nodes": [[["Encoder_Hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Latent", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Latent", "inbound_nodes": [[["Encoder_Hidden2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Latent", 0, 0]]}, "name": "encoder", "inbound_nodes": [[["input_34", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_33"}, "name": "input_33", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "dnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RnaEncoded"}, "name": "RnaEncoded", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "DrugFingerprintInput"}, "name": "DrugFingerprintInput", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_16", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_16", "inbound_nodes": [[["RnaEncoded", 0, 0, {}], ["DrugFingerprintInput", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden1", "inbound_nodes": [[["concatenate_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_32", "inbound_nodes": [[["Hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden2", "inbound_nodes": [[["dropout_32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_33", "inbound_nodes": [[["Hidden2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Output", "inbound_nodes": [[["dropout_33", 0, 0, {}]]]}], "input_layers": [["RnaEncoded", 0, 0], ["DrugFingerprintInput", 0, 0]], "output_layers": [["Output", 0, 0]]}, "name": "dnn", "inbound_nodes": [[["encoder", 1, 0, {}], ["input_33", 0, 0, {}]]]}], "input_layers": [["input_34", 0, 0], ["input_33", 0, 0]], "output_layers": [["dnn", 1, 0]]}, "shared_object_id": 29, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 463]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 256]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 463]}, {"class_name": "TensorShape", "items": [null, 256]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 463]}, "float32", "input_34"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 256]}, "float32", "input_33"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 463]}, "float32", "input_34"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 256]}, "float32", "input_33"]}], "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_34"}, "name": "input_34", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden2", "inbound_nodes": [[["Encoder_Hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Latent", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Latent", "inbound_nodes": [[["Encoder_Hidden2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Latent", 0, 0]]}, "name": "encoder", "inbound_nodes": [[["input_34", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_33"}, "name": "input_33", "inbound_nodes": [], "shared_object_id": 13}, {"class_name": "Functional", "config": {"name": "dnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RnaEncoded"}, "name": "RnaEncoded", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "DrugFingerprintInput"}, "name": "DrugFingerprintInput", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_16", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_16", "inbound_nodes": [[["RnaEncoded", 0, 0, {}], ["DrugFingerprintInput", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden1", "inbound_nodes": [[["concatenate_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_32", "inbound_nodes": [[["Hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden2", "inbound_nodes": [[["dropout_32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_33", "inbound_nodes": [[["Hidden2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Output", "inbound_nodes": [[["dropout_33", 0, 0, {}]]]}], "input_layers": [["RnaEncoded", 0, 0], ["DrugFingerprintInput", 0, 0]], "output_layers": [["Output", 0, 0]]}, "name": "dnn", "inbound_nodes": [[["encoder", 1, 0, {}], ["input_33", 0, 0, {}]]], "shared_object_id": 28}], "input_layers": [["input_34", 0, 0], ["input_33", 0, 0]], "output_layers": [["dnn", 1, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_34", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_34"}}2
�-root.layer_with_weights-0"_tf_keras_network*�-{"name": "encoder", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden1", "inbound_nodes": [[["Input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden2", "inbound_nodes": [[["Encoder_Hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Latent", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Latent", "inbound_nodes": [[["Encoder_Hidden2", 0, 0, {}]]]}], "input_layers": [["Input", 0, 0]], "output_layers": [["Latent", 0, 0]]}, "inbound_nodes": [[["input_34", 0, 0, {}]]], "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 463]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 463]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 463]}, "float32", "Input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 463]}, "float32", "Input"]}, "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input"}, "name": "Input", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden1", "inbound_nodes": [[["Input", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "Encoder_Hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Encoder_Hidden2", "inbound_nodes": [[["Encoder_Hidden1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "Latent", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Latent", "inbound_nodes": [[["Encoder_Hidden2", 0, 0, {}]]], "shared_object_id": 11}], "input_layers": [["Input", 0, 0]], "output_layers": [["Latent", 0, 0]]}}}2
�root.layer-2"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_33", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_33"}}2
�>root.layer_with_weights-1"_tf_keras_network*�>{"name": "dnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "dnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RnaEncoded"}, "name": "RnaEncoded", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "DrugFingerprintInput"}, "name": "DrugFingerprintInput", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_16", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_16", "inbound_nodes": [[["RnaEncoded", 0, 0, {}], ["DrugFingerprintInput", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden1", "inbound_nodes": [[["concatenate_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_32", "inbound_nodes": [[["Hidden1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden2", "inbound_nodes": [[["dropout_32", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_33", "inbound_nodes": [[["Hidden2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Ones", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Output", "inbound_nodes": [[["dropout_33", 0, 0, {}]]]}], "input_layers": [["RnaEncoded", 0, 0], ["DrugFingerprintInput", 0, 0]], "output_layers": [["Output", 0, 0]]}, "inbound_nodes": [[["encoder", 1, 0, {}], ["input_33", 0, 0, {}]]], "shared_object_id": 28, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 256]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 256]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32]}, "float32", "RnaEncoded"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 256]}, "float32", "DrugFingerprintInput"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32]}, "float32", "RnaEncoded"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 256]}, "float32", "DrugFingerprintInput"]}], "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RnaEncoded"}, "name": "RnaEncoded", "inbound_nodes": [], "shared_object_id": 14}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "DrugFingerprintInput"}, "name": "DrugFingerprintInput", "inbound_nodes": [], "shared_object_id": 15}, {"class_name": "Concatenate", "config": {"name": "concatenate_16", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_16", "inbound_nodes": [[["RnaEncoded", 0, 0, {}], ["DrugFingerprintInput", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden1", "inbound_nodes": [[["concatenate_16", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_32", "inbound_nodes": [[["Hidden1", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden2", "inbound_nodes": [[["dropout_32", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_33", "inbound_nodes": [[["Hidden2", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Output", "inbound_nodes": [[["dropout_33", 0, 0, {}]]], "shared_object_id": 27}], "input_layers": [["RnaEncoded", 0, 0], ["DrugFingerprintInput", 0, 0]], "output_layers": [["Output", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "clipvalue": 5, "learning_rate": {"class_name": "ExponentialDecay", "config": {"initial_learning_rate": 0.001, "decay_steps": 5, "decay_rate": 0.96, "staircase": false, "name": null}, "shared_object_id": 35}, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�
�	.root.layer_with_weights-0.layer_with_weights-0"_tf_keras_layer*�{"name": "Encoder_Hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Encoder_Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Input", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 463}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 463]}}2
�	.root.layer_with_weights-0.layer_with_weights-1"_tf_keras_layer*�{"name": "Encoder_Hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Encoder_Hidden2", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Encoder_Hidden1", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�	.root.layer_with_weights-0.layer_with_weights-2"_tf_keras_layer*�{"name": "Latent", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Latent", "trainable": true, "dtype": "float32", "units": 32, "activation": {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Encoder_Hidden2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�!root.layer_with_weights-1.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "RnaEncoded", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RnaEncoded"}}2
�!root.layer_with_weights-1.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "DrugFingerprintInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "DrugFingerprintInput"}}2
�!root.layer_with_weights-1.layer-2"_tf_keras_layer*�{"name": "concatenate_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_16", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["RnaEncoded", 0, 0, {}], ["DrugFingerprintInput", 0, 0, {}]]], "shared_object_id": 16, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 256]}]}2
�.root.layer_with_weights-1.layer_with_weights-0"_tf_keras_layer*�{"name": "Hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_16", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 288}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 288]}}2
�!root.layer_with_weights-1.layer-4"_tf_keras_layer*�{"name": "dropout_32", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_32", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Hidden1", 0, 0, {}]]], "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}2
�.root.layer_with_weights-1.layer_with_weights-1"_tf_keras_layer*�{"name": "Hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_32", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}2
�!root.layer_with_weights-1.layer-6"_tf_keras_layer*�{"name": "dropout_33", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_33", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Hidden2", 0, 0, {}]]], "shared_object_id": 24, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}2
� .root.layer_with_weights-1.layer_with_weights-2"_tf_keras_layer*�{"name": "Output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_33", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}2
�;9root.layer_with_weights-0.layer_with_weights-0.activation"_tf_keras_layer*�{"name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 2}2
��-root.layer_with_weights-1.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 42}2