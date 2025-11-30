# System Verification & Testing Guide

## Quick Test

Run the comprehensive test suite:

```bash
python test_system.py
```

This will verify all components and run a short 10-step integration test.

## Test Coverage

The test suite includes:

1. **Import Verification** - All modules can be imported
2. **PyTorch Setup** - PyTorch is installed and working
3. **World Component** - Environment and energy calculation
4. **Neuro-Chemical Engine** - Dopamine/Serotonin dynamics
5. **Graph Attention Network** - GAT reasoning with soul vectors
6. **Action Decoder** - Action selection from latent states
7. **JEPA Predictor** - World model prediction
8. **Integrated System** - Full 10-step execution

## Expected Output

Successful test run should show:

```
############################################################
# ADVANCED AI META-COGNITION SYSTEM
# COMPREHENSIVE TEST SUITE
############################################################

============================================================
TEST 1: IMPORT VERIFICATION
============================================================
✅ All core modules imported successfully

============================================================
TEST 2: PYTORCH SETUP
============================================================
PyTorch Version: 2.x.x
CUDA Available: True/False
Device: cuda/cpu
Matrix multiplication test: torch.Size([3, 3]) tensor created
✅ PyTorch setup verified

[... more tests ...]

============================================================
TEST 8: INTEGRATED SYSTEM (10 STEPS)
============================================================
Initializing system...
Running 10 steps...

Step 001 | CHAOS | D:0.50 S:0.50 | E:0.XXXX C:0.XXXX | ...
[... 10 steps ...]

[Summary Statistics]
  Total steps: 10
  Energy reduction: X.XXXX
  Final consistency: X.XXXX

✅ Integrated system working

############################################################
# TEST SUMMARY
############################################################
✅ PASS: Import Verification
✅ PASS: PyTorch Setup
✅ PASS: World Component
✅ PASS: Neuro-Chemical Engine
✅ PASS: Graph Attention Network
✅ PASS: Action Decoder
✅ PASS: JEPA Predictor
✅ PASS: Integrated System

Total: 8/8 tests passed

🎉 ALL TESTS PASSED! System is ready for experiments.
############################################################
```

## Manual Testing

### Test 1: Basic System (Original)

```bash
python main_system.py
```

Should run 50 steps and show energy reduction.

### Test 2: Enhanced System (Default Config)

```bash
python main_system_enhanced.py --steps 30 --seed 42
```

Should run 30 steps with reproducible results.

### Test 3: Experimental Config

```bash
python main_system_enhanced.py --config experimental --steps 50 --save-metrics test_run.json
```

Should run with larger network and save metrics.

### Test 4: Analyze Results

```bash
python evaluation.py test_run.json
```

Should print comprehensive analysis report.

## Performance Benchmarks

### Expected Performance (CPU)

- **Import time**: < 5 seconds
- **Initialization**: < 2 seconds
- **Step time**: 50-200ms per step
- **10 steps**: 1-3 seconds
- **100 steps**: 10-30 seconds

### Expected Performance (CUDA)

- **Import time**: < 5 seconds
- **Initialization**: < 3 seconds (GPU warmup)
- **Step time**: 10-50ms per step
- **10 steps**: < 1 second
- **100 steps**: 2-10 seconds

## Troubleshooting

### Issue: Import errors

**Solution:**
```bash
pip install torch numpy --upgrade
```

### Issue: CUDA not available

**Normal** - System will use CPU automatically. To use GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Test fails at step X

**Check:**
1. All files present in repository
2. Python version >= 3.8
3. Dependencies installed correctly

### Issue: NaN values in output

**Likely cause:** Numerical instability

**Solution:**
```bash
python main_system_enhanced.py --config default --seed 42
# Default config includes gradient clipping and stable learning rate
```

### Issue: Slow execution

**Solution:**
1. Use smaller latent dimensions
2. Use CPU if GPU is old
3. Reduce max_steps

```python
from config import get_default_config
config = get_default_config()
config.architecture.latent_dim = 4  # Smaller
config.training.max_steps = 50      # Fewer steps
```

## Verification Checklist

- [ ] All 8 tests pass
- [ ] No import errors
- [ ] PyTorch detects correct device
- [ ] Energy decreases over steps
- [ ] Dopamine/Serotonin dynamics work
- [ ] Mode transitions CHAOS → ORDER
- [ ] Grid density increases from 0
- [ ] No NaN or inf values
- [ ] Metrics save correctly
- [ ] Analysis tools work

## Next Steps After Testing

1. **Run longer experiments:**
   ```bash
   python main_system_enhanced.py --config experimental --steps 500 --save-metrics long_run.json
   ```

2. **Compare configurations:**
   ```bash
   python main_system_enhanced.py --config default --save-metrics default.json
   python main_system_enhanced.py --config experimental --save-metrics exp.json
   python evaluation.py default.json exp.json
   ```

3. **Custom experiments:**
   - Modify `config.py` parameters
   - Test different world sizes
   - Adjust neuro-chemical thresholds
   - Experiment with curriculum settings

4. **Analysis:**
   - Use `evaluation.py` to analyze convergence
   - Check phase transitions
   - Study hormone correlations
   - Examine action distributions

## Success Criteria

✅ System is working correctly if:

1. All tests pass (8/8)
2. Energy reduces over time (typically 40-80% reduction)
3. System transitions from CHAOS to ORDER
4. Grid density reaches ~10% (boredom penalty target)
5. No crashes or numerical errors
6. Metrics can be saved and analyzed

If all criteria are met, the system is **ready for research experiments**!
