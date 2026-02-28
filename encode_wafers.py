#!/usr/bin/env python
"""
Latent Space Generation Script

Usage:
    python run_latent_space_with_events.py

This script will:
1. Load all ROOT files matching the pattern
2. Process them through the encoder
3. Save latent space with event indices preserved
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Concatenate
from qkeras import QActivation, QConv2D, QDense, quantized_bits
import uproot
import awkward as ak
import glob
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define eLink configurations and their corresponding weight files
ELINK_CONFIGS = {
    2: {
        'weights_file': 'encoders/encoder_model_NoBiasModel_elink_2.hdf5',
        'layer_mask': lambda layers: (layers < 7) | (layers > 13),
        'description': 'Layers < 7 or > 13'
    },
    3: {
        'weights_file': 'encoders/encoder_model_NoBiasModel_elink_3.hdf5',
        'layer_mask': lambda layers: layers == 13,
        'description': 'Layer 13'
    },
    4: {
        'weights_file': 'encoders/encoder_model_NoBiasModel_elink_4.hdf5',
        'layer_mask': lambda layers: (layers == 7) | (layers == 11),
        'description': 'Layers 7 or 11'
    },
    5: {
        'weights_file': 'encoders/encoder_model_NoBiasModel_elink_5.hdf5',
        'layer_mask': lambda layers: (layers <= 11) & (layers >= 5),
        'description': 'Layers 5-11'
    },
}

CONFIG = {
    'root_file_pattern': 'data/output_Phase2_HGCalL1T_Clustering_*.root',
    'eLinks_to_process': [2, 3, 4, 5],  # Process all eLink configurations
    'batch_size': 1024,
    'file_limit': -1,  # -1 means process all files
    'output_dir': 'data',  # Directory for output files
}

# Set random seed for reproducibility
np.random.seed(42)
tf.config.run_functions_eagerly(True)

# ============================================================================
# CUSTOM KERAS LAYERS
# ============================================================================

class KerasPaddingLayer(tf.keras.layers.Layer):
    """Add zero border to match CAE geometry"""
    def call(self, x):
        padding = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        return tf.pad(x, padding, mode='CONSTANT', constant_values=0)

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        return (batch_size, height + 1, width + 1, channels)


class KerasMinimumLayer(tf.keras.layers.Layer):
    """Limit values to maximum threshold"""
    def __init__(self, saturation_value=1, **kwargs):
        super(KerasMinimumLayer, self).__init__(**kwargs)
        self.saturation_value = saturation_value

    def call(self, x):
        return tf.minimum(x, self.saturation_value)

    def compute_output_shape(self, input_shape):
        return input_shape


class KerasFloorLayer(tf.keras.layers.Layer):
    """Force values to nearest integer"""
    def call(self, x):
        return tf.math.floor(x)

    def compute_output_shape(self, input_shape):
        return input_shape


# ============================================================================
# ENCODER MODEL
# ============================================================================

def build_encoder_model(eLinks=2):
    """Build quantized encoder model"""
    input_shape = (8, 8, 1) # wafer input
    condition_shape = (8,)  # condition input

    # Map eLinks to quantization bits
    bits_per_outputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    bits_per_output = bits_per_outputLink[eLinks]

    # Encoder inputs
    wafer_input = Input(shape=input_shape, name='Wafer_Input')
    condition_input = Input(shape=condition_shape, name='Condition_Input')

    # Encoder backbone
    x = QActivation(activation=quantized_bits(bits=8, integer=1), name='Input_Quantization')(wafer_input)
    x = KerasPaddingLayer(name='Pad')(x)
    x = QConv2D(
        filters=8, kernel_size=3, strides=2, padding='valid',
        kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
        bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
        name='Conv2D'
    )(x)
    x = QActivation(activation=quantized_bits(bits=8, integer=1), name='Activation')(x)
    x = Flatten(name='Flat')(x)
    x = QDense(
        units=16,
        kernel_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
        bias_quantizer=quantized_bits(bits=6, integer=0, keep_negative=1, alpha=1),
        name='Dense_Layer'
    )(x)
    x = QActivation(activation=quantized_bits(bits=9, integer=1), name='Latent_Quantization')(x)
    latent_output = x

    # Fixed point packing
    if bits_per_output > 0:
        n_integer_bits = 1
        n_decimal_bits = bits_per_output - n_integer_bits
        output_max_int_size = 1 << n_decimal_bits
        output_saturation_value = (1 << n_integer_bits) - 1. / (1 << n_decimal_bits)

        latent_output = KerasFloorLayer(name='Latent_Floor')(latent_output * output_max_int_size)
        latent_output = KerasMinimumLayer(saturation_value=output_saturation_value, name='Latent_Clip')(latent_output / output_max_int_size)

    # Concatenate latent with conditions (16 + 8 = 24D)
    latent_output = Concatenate(axis=1, name='Concat')([latent_output, condition_input])
    encoder = Model(inputs=[wafer_input, condition_input], outputs=latent_output, name='Encoder_Model')

    return encoder


# ============================================================================
# DATA LOADING
# ============================================================================

def load_root_file(file_path, selected_eLinks=-1):
    """
    Load ROOT file and return inputs, conditions, and event indices

    Returns:
        inputs: (N_wafers, 8, 8) numpy array
        conditions: (N_wafers, 8) numpy array
        event_indices: (N_wafers,) numpy array - which event each wafer belongs to
        file_indices: (N_wafers,) numpy array - which file each wafer came from
    """

    all_inputs, all_conditions, all_event_indices, all_file_indices = [], [], [], []
    tree_name = 'Events'
    global_event_counter = 0

    print(f"Processing: {file_path}")

    try:
        with uproot.open(file_path) as root_file:
            if tree_name not in root_file:
                raise ValueError(f"Tree '{tree_name}' not found in '{file_path}'")
            tree = root_file[tree_name]

            # Define branches to read
            branches = [
                "GenPart_pt", "L1THGCAL_wafer_layer", "L1THGCAL_wafer_eta",
                "L1THGCAL_wafer_waferv", "L1THGCAL_wafer_waferu", "L1THGCAL_wafer_wafertype"
            ]
            branches.extend([f"L1THGCAL_wafer_CALQ_{j}" for j in range(64)])
            branches.extend([f"L1THGCAL_wafer_AEin_{j}" for j in range(64)])

            data = tree.arrays(branches, library="ak")
            n_events = len(data['GenPart_pt'])

            print(f"    Events in file: {n_events}")

            # Track wafers per event
            wafers_per_event = ak.num(data["L1THGCAL_wafer_layer"])
            total_wafers_before_filter = ak.sum(wafers_per_event)

            # Create event indices
            event_indices = []
            file_indices = []
            for event_idx, n_wafers in enumerate(wafers_per_event):
                event_indices.extend([global_event_counter + event_idx] * n_wafers)
                file_indices.extend([file_index] * n_wafers)

            event_indices = np.array(event_indices)
            file_indices = np.array(file_indices)

            # Flatten data and normalize
            layers = ak.to_numpy(ak.flatten(data["L1THGCAL_wafer_layer"]))
            eta = ak.to_numpy(ak.flatten(data["L1THGCAL_wafer_eta"])) / 3.1
            wafer_v = ak.to_numpy(ak.flatten(data["L1THGCAL_wafer_waferv"])) / 12
            wafer_u = ak.to_numpy(ak.flatten(data["L1THGCAL_wafer_waferu"])) / 12
            wafer_type = ak.to_numpy(ak.flatten(data["L1THGCAL_wafer_wafertype"])).astype(int)
            one_hot_wafertype = np.eye(np.max(wafer_type) + 1)[wafer_type]

            sum_CALQ = np.sum([ak.to_numpy(ak.flatten(data[f"L1THGCAL_wafer_CALQ_{j}"])) for j in range(64)], axis=0)
            sum_CALQ = np.log(sum_CALQ + 1)

            inputs = np.stack([ak.to_numpy(ak.flatten(data[f"L1THGCAL_wafer_AEin_{j}"])) for j in range(64)], axis=-1)
            inputs = np.reshape(inputs, (-1, 8, 8))

            # Apply eLinks mask using the configuration
            if selected_eLinks in ELINK_CONFIGS:
                selection_mask = ELINK_CONFIGS[selected_eLinks]['layer_mask'](layers)
            else:
                raise ValueError(f"Unknown eLinks configuration: {selected_eLinks}. Available: {list(ELINK_CONFIGS.keys())}")

            # Apply mask to all arrays
            inputs = inputs[selection_mask]
            eta = eta[selection_mask]
            wafer_v = wafer_v[selection_mask]
            wafer_u = wafer_u[selection_mask]
            one_hot_wafertype = one_hot_wafertype[selection_mask]
            sum_CALQ = sum_CALQ[selection_mask]
            layers_normalized = (layers[selection_mask] - 1) / 46
            event_indices = event_indices[selection_mask]
            file_indices = file_indices[selection_mask]

            print(f"    Wafers: {total_wafers_before_filter} → {inputs.shape[0]} (after eLinks={selected_eLinks} filter)")

            # Build condition features
            conditions = np.hstack([
                eta[:, np.newaxis], wafer_v[:, np.newaxis], wafer_u[:, np.newaxis],
                one_hot_wafertype, sum_CALQ[:, np.newaxis], layers_normalized[:, np.newaxis]
            ])

            all_inputs.append(inputs)
            all_conditions.append(conditions)
            all_event_indices.append(event_indices)
            all_file_indices.append(file_indices)

            global_event_counter += n_events

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

    if not all_inputs:
        raise ValueError("No data loaded from any files!")

    # Concatenate all data
    inputs_final = np.concatenate(all_inputs)
    conditions_final = np.concatenate(all_conditions)
    event_indices_final = np.concatenate(all_event_indices)
    file_indices_final = np.concatenate(all_file_indices)

    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Total wafers loaded: {len(inputs_final):,}")
    print(f"  Total events: {global_event_counter:,}")
    print(f"  Total files: {len(files)}")
    print(f"  Avg wafers per event: {len(inputs_final) / global_event_counter:.1f}")
    print(f"{'='*70}\n")

    return inputs_final, conditions_final, event_indices_final, file_indices_final


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_single_file(root_file_path, eLinks_to_process, batch_size, output_dir):
    """
    Process a single ROOT file with all eLink configurations

    Saves one NPZ file per input ROOT file with merged latent spaces from all eLinks
    """
    root_filename = Path(root_file_path).stem
    output_file = Path(output_dir) / f"{root_filename}_latent.npz"

    print("\n" + "="*70)
    print(f"PROCESSING FILE: {Path(root_file_path).name}")
    print("="*70)

    # Lists to collect data from all eLinks
    all_latents = []
    all_conditions = []
    all_event_indices = []
    all_elink_ids = []

    for eLink_id in eLinks_to_process:
        if eLink_id not in ELINK_CONFIGS:
            print(f"\n⚠️  WARNING: Unknown eLink {eLink_id}, skipping...")
            continue

        config = ELINK_CONFIGS[eLink_id]
        weights_file = config['weights_file']

        print(f"\n--- Processing eLink {eLink_id} ({config['description']}) ---")

        # Check weights
        if not Path(weights_file).exists():
            print(f"❌ ERROR: Weights file not found: {weights_file}")
            print(f"Skipping eLink {eLink_id}...")
            continue

        # Build and load encoder
        print(f"Building encoder for eLink {eLink_id}...")
        encoder = build_encoder_model(eLinks=eLink_id)
        encoder.load_weights(weights_file)
        print(f"✓ Loaded weights: {weights_file}")

        # Load data with eLink-specific filter
        print(f"Loading data with eLink {eLink_id} filter...")
        inputs, conditions, event_indices, file_indices = load_root_file(
            file_path=root_file_path,  # Process only this specific file
            selected_eLinks=eLink_id
        )

        # Generate latent space
        print(f"Generating latent space for {len(inputs):,} wafers...")
        wafer_tensor = inputs[..., np.newaxis].astype("float32")
        cond_tensor = conditions.astype("float32")

        encoder_output = encoder.predict(
            [wafer_tensor, cond_tensor],
            batch_size=batch_size,
            verbose=0
        )

        latent_only = encoder_output[:, :16].astype("float32")

        # Collect results
        all_latents.append(latent_only)
        all_conditions.append(conditions.astype("float32"))
        all_event_indices.append(event_indices.astype("int32"))
        all_elink_ids.append(np.full(len(inputs), eLink_id, dtype="int32"))

        print(f"✓ eLink {eLink_id}: {len(inputs):,} wafers processed")

    if not all_latents:
        print(f"\n❌ ERROR: No eLink data generated for {root_file_path}")
        return None

    # Merge all eLink data
    print(f"\nMerging data from all eLinks...")
    merged_latent = np.concatenate(all_latents)
    merged_conditions = np.concatenate(all_conditions)
    merged_event_indices = np.concatenate(all_event_indices)
    merged_elink_ids = np.concatenate(all_elink_ids)

    # Save merged output
    print(f"Saving merged output to: {output_file}")
    np.savez_compressed(
        output_file,
        latent=merged_latent,
        conditions=merged_conditions,
        event_index=merged_event_indices,
        elink_id=merged_elink_ids
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"SAVED: {output_file.name}")
    print(f"{'='*70}")
    print(f"  Total wafers: {len(merged_latent):,}")
    print(f"  Arrays:")
    print(f"    latent:       {merged_latent.shape}")
    print(f"    conditions:   {merged_conditions.shape}")
    print(f"    event_index:  {merged_event_indices.shape}")
    print(f"    elink_id:     {merged_elink_ids.shape}")

    print(f"\n  Per-eLink breakdown:")
    for eLink_id in eLinks_to_process:
        n_wafers = np.sum(merged_elink_ids == eLink_id)
        if n_wafers > 0:
            print(f"    eLink {eLink_id}: {n_wafers:,} wafers")

    print(f"\n  File size: {output_file.stat().st_size / 1024**2:.1f} MB")
    print(f"{'='*70}\n")

    return str(output_file)


def main():
    """Main execution function"""

    print("\n" + "="*70)
    print("LATENT SPACE GENERATION - MULTI-ELINK PER FILE")
    print("="*70)

    eLinks_to_process = CONFIG['eLinks_to_process']

    print(f"\nConfiguration:")
    print(f"  ROOT file pattern: {CONFIG['root_file_pattern']}")
    print(f"  eLinks to process: {eLinks_to_process}")
    print(f"  File limit: {CONFIG['file_limit'] if CONFIG['file_limit'] > 0 else 'All files'}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Output directory: {CONFIG['output_dir']}")

    print(f"\neLink configurations:")
    for eLink_id in eLinks_to_process:
        if eLink_id in ELINK_CONFIGS:
            config = ELINK_CONFIGS[eLink_id]
            print(f"  eLink {eLink_id}: {config['description']}")
            print(f"    Weights: {config['weights_file']}")
            print(f"    Available: {'✓' if Path(config['weights_file']).exists() else '❌'}")

    # Find all ROOT files
    root_files = sorted(glob.glob(CONFIG['root_file_pattern']))
    if CONFIG['file_limit'] > 0:
        root_files = root_files[:CONFIG['file_limit']]

    if not root_files:
        print(f"\n❌ ERROR: No ROOT files found matching: {CONFIG['root_file_pattern']}")
        sys.exit(1)

    print(f"\nFound {len(root_files)} ROOT file(s) to process")

    # Process each ROOT file
    output_files = []

    for file_idx, root_file in enumerate(root_files):
        print(f"\n{'#'*70}")
        print(f"FILE {file_idx + 1}/{len(root_files)}")
        print(f"{'#'*70}")

        try:
            output_file = process_single_file(
                root_file_path=root_file,
                eLinks_to_process=eLinks_to_process,
                batch_size=CONFIG['batch_size'],
                output_dir=CONFIG['output_dir']
            )

            if output_file:
                output_files.append(output_file)

        except Exception as e:
            print(f"\n❌ ERROR processing {Path(root_file).name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nContinuing with next file...\n")
            continue

    # Final summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)

    if output_files:
        print(f"\n✓ Successfully processed {len(output_files)}/{len(root_files)} file(s):")
        for f in output_files:
            print(f"  - {Path(f).name}")


if __name__ == "__main__":
    main()
