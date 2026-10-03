# Real-Data Validation

## Goal

Validate whether the same compact Transformer pipeline behaves differently on a real public randomness stream versus a controlled weak source with known temporal dependence.

The objective is **not** to maximize accuracy on a high-quality randomness source. A well-behaved next-bit predictor should remain near chance on such data. The controlled weak source provides the positive-control case where prediction should improve because exploitable structure is intentionally present.

## Real source: NIST Randomness Beacon 2.0

Data collection used 2,000 public NIST Randomness Beacon 2.0 pulses. Each pulse contributed 512 output bits, yielding:

- Pulses: **2,000**
- Bits per pulse: **512**
- Total bytes: **128,000**
- Total bits: **1,024,000**

The bitstream was stored locally as `data/nist_beacon.bin` and evaluated through the existing raw-binary input path.

### NIST experiment

```bash
python scripts/run_experiment.py \
  --input data/nist_beacon.bin \
  --format raw \
  --context-length 64 \
  --target-bits 1 \
  --epochs 10 \
  --batch-size 256 \
  --output reports/nist_beacon_gpu
```

### NIST held-out test result

| Metric | Result |
|---|---:|
| Accuracy | **0.5004038** |
| Balanced accuracy | **0.5000000** |
| ROC-AUC | **0.5004317** |
| Cross-entropy bits/target | **1.0000024** |
| Perplexity | **2.0000033** |
| Brier score | **0.5000017** |
| Mean max probability | **0.5013973** |
| Predictive min-entropy proxy | **0.9959738 bits/bit** |
| Held-out samples | **153,536** |
| Majority baseline | **0.5004038** |

Internal research diagnostics reported:

```text
failed_tests: []
```

Interpretation: the Transformer did not outperform a chance-level or majority-class predictor on the real beacon sequence. ROC-AUC was essentially 0.5 and cross-entropy was essentially 1 bit per predicted bit.

This does **not** prove cryptographic security, physical randomness, or compliance with an official NIST test standard.

## Controlled positive control: first-order Markov source

The repository's built-in Markov generator uses a persistence probability of approximately 0.72, creating deliberate next-bit temporal dependence.

### Markov experiment

```bash
python scripts/run_experiment.py \
  --source markov \
  --num-bits 1024000 \
  --seed 42 \
  --context-length 128 \
  --target-bits 1 \
  --d-model 96 \
  --nhead 8 \
  --num-layers 3 \
  --epochs 20 \
  --batch-size 256 \
  --output reports/markov_gpu
```

### Markov held-out test result

| Metric | Result |
|---|---:|
| Accuracy | **0.7200467** |
| Balanced accuracy | **0.7200462** |
| ROC-AUC | **0.7205943** |
| Cross-entropy bits/target | **0.8556042** |
| Perplexity | **1.8095164** |
| Brier score | **0.4032816** |
| Mean max probability | **0.7124532** |
| Predictive min-entropy proxy | **0.4891329 bits/bit** |
| Held-out samples | **153,472** |
| Majority baseline | **0.5006451** |

Internal research diagnostics flagged:

```text
block_frequency
runs
serial_pair
autocorrelation_lag_1
```

Interpretation: the same modeling pipeline successfully exploited the intentional temporal dependence in the controlled Markov source, raising next-bit accuracy from a roughly 50% majority baseline to approximately **72%**.

## Comparison

| Source | Accuracy | ROC-AUC | Cross-Entropy | Majority baseline |
|---|---:|---:|---:|---:|
| NIST Beacon real data | **50.04%** | **0.5004** | **1.0000** | **50.04%** |
| Controlled Markov source | **72.00%** | **0.7206** | **0.8556** | **50.06%** |

The comparison demonstrates that the pipeline can distinguish between a source with exploitable temporal structure and a real public beacon stream where the trained next-bit predictor remains at chance level.

## Compute

The final validation runs were executed with a CUDA-enabled PyTorch environment on an **NVIDIA RTX 2000 Ada Generation** GPU. GPU acceleration changed runtime, not the intended interpretation of the results.

## Limitations

- Only one real public source was evaluated in this validation stage.
- The NIST Beacon result should not be interpreted as a security proof.
- The Markov source is synthetic and intentionally weak.
- The repository's internal statistical diagnostics are research checks, not official NIST SP 800-22 or SP 800-90B implementations.
- Additional repeated-seed runs and other real sources would improve robustness of the empirical comparison.

## Safe portfolio statement

> Built and validated a compact Transformer-based randomness analysis pipeline on more than 1 million real NIST Randomness Beacon bits and controlled weak generators. The model remained at chance level on NIST data (50.0% accuracy, ROC-AUC 0.500) while reaching 72.0% accuracy and 0.721 ROC-AUC on a temporally correlated Markov source, demonstrating sensitivity to predictable structure without falsely detecting predictability in the real beacon stream.
