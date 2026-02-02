# Quick Reference Checklist

## 📋 Seven Issues at a Glance

### 🔴 CRITICAL - Must Fix for Major Improvement

- [ ] **Issue 1: Soft-Min Loss (Line 75 of fl_train.py)**
  - Symptom: FL uses soft-min, centralized uses hard-min
  - Fix: Replace lines 71-76 with hard-min
  - Expected improvement: +10-15%
  - Time to fix: 5 minutes
  
- [ ] **Issue 2: Bottleneck Indicator Dimension (Line 318 of fl_train.py)**
  - Symptom: Bottleneck is [B, K_AP] instead of [B, K_UE]
  - Fix: Reshape rate to be per-UE before softmax
  - Expected improvement: +5-10%
  - Time to fix: 10 minutes

- [ ] **Issue 3: Aggregated Interference (Line 349 of fl_train.py)**
  - Symptom: Using mean/max instead of full per-UE information
  - Fix: Don't aggregate over UEs dimension
  - Expected improvement: +10-15%
  - Time to fix: 20 minutes

### 🟠 MAJOR - Should Fix for Noticeable Improvement

- [ ] **Issue 4: Gradient Flow Asymmetry (Lines 99-122 of comm.py)**
  - Symptom: Each AP only learns from its own DS, not how it affects others
  - Fix: Redesign to feed back interference impact gradients
  - Expected improvement: +5-10%
  - Time to fix: 1 hour
  
- [ ] **Issue 5: Information Staleness (Lines 104-128 of fl_train.py)**
  - Symptom: DS/PC/UI from start of round, but model updates each epoch
  - Fix: Recompute DS/PC/UI locally at each epoch
  - Expected improvement: +3-7%
  - Time to fix: 20 minutes

### 🟡 MODERATE - Nice to Have

- [ ] **Issue 6: Circular Augmented Features (Line 371 of fl_train.py)**
  - Symptom: global_sinr fed back as input from current sample
  - Fix: Use previous round's global_sinr or remove entirely
  - Expected improvement: +2-5%
  - Time to fix: 10 minutes

- [ ] **Issue 7: Model Architecture (GNN.py lines 308-514)**
  - Symptom: Complex architecture without better information
  - Fix: Simplify or redesign for decentralized learning
  - Expected improvement: +1-3%
  - Time to fix: 2+ hours

---

## 🎯 Recommended Fix Order

### Phase 1: Quick Wins (1 hour)
1. Fix Issue 1 (Soft-Min) → Test
2. Fix Issue 5 (Staleness) → Test  
3. Fix Issue 6 (Circular features) → Test

**Expected result after Phase 1:** 1.3 → 1.38-1.42 (+3-8%)

### Phase 2: Information Fixes (1.5 hours)
1. Fix Issue 2 (Bottleneck dimension) → Test
2. Fix Issue 3 (Aggregated interference) → Test

**Expected result after Phase 2:** 1.38 → 1.50-1.58 (+8-15%)

### Phase 3: Fundamental Redesign (2+ hours)
1. Fix Issue 4 (Gradient asymmetry) → Test
2. Redesign Issue 7 (Architecture) → Test

**Expected result after Phase 3:** 1.50 → 1.65+ (+10%+)

---

## 🔍 How to Validate Each Fix

### Quick Validation Protocol

```bash
# Baseline (current code)
python FlGrad.py --num_rounds 50 --fl_scheme fedavg --eval_plot | tee baseline.log

# After each fix:
python FlGrad.py --num_rounds 50 --fl_scheme fedavg --eval_plot | tee fix_N.log

# Compare results in logs:
# Look for line: "Avg Eval Rate = X.XXXX"
# Compare final X.XXXX values
```

### Expected Baseline Result
```
===Training FL-GNN using FEDAVG...
...
Round 050: Avg Eval Rate = 1.3XXX
```

### Expected After Issue 1 Fix
```
Round 050: Avg Eval Rate = 1.40-1.45
```

---

## 📊 Performance Tracking Table

| Issue | Status | Baseline | After Fix | Net Gain | Cumulative |
|-------|--------|----------|-----------|----------|-----------|
| Baseline | ☐ | 1.3000 | — | — | 1.3000 |
| Issue 1 (Soft-min) | ☐ | 1.3000 | 1.4000 | +7.7% | 1.4000 |
| Issue 2 (Bottleneck) | ☐ | 1.4000 | 1.4600 | +4.3% | 1.4600 |
| Issue 3 (Info) | ☐ | 1.4600 | 1.5500 | +6.2% | 1.5500 |
| Issue 4 (Gradient) | ☐ | 1.5500 | 1.6200 | +4.5% | 1.6200 |
| Issue 5 (Staleness) | ☐ | 1.6200 | 1.6400 | +1.2% | 1.6400 |
| Issue 6 (Circular) | ☐ | 1.6400 | 1.6500 | +0.6% | 1.6500 |
| **All fixes** | ☐ | **1.3000** | **1.6500** | **+26.9%** | **1.6500** |

---

## 🚀 Implementation Checklist

### Issue 1: Soft-Min → Hard-Min
- [ ] Open `Utils/fl_train.py`
- [ ] Find line 71: `min_rate, _ = torch.min(rate.detach(), dim=1)`
- [ ] Delete lines 73-76 (soft-min calculation)
- [ ] Change line 77 to: `loss = -min_rate.mean()`
- [ ] Save and test

### Issue 2: UE-Level Bottleneck
- [ ] Open `Utils/fl_train.py`
- [ ] Find line 318-319: global_rate calculation
- [ ] Add this BEFORE bottleneck_indicator:
  ```python
  # Convert from [B, K_AP, K_UE] to per-UE rate
  sum_DS_per_ue = all_DS_stack.sum(dim=1)  # [B, K_UE]
  sum_PC_per_ue = all_PC_stack.sum(dim=1)
  sum_UI_per_ue = all_UI_stack.sum(dim=1)
  global_rate = rate_from_component(sum_DS_per_ue, sum_PC_per_ue, sum_UI_per_ue, num_antenna)
  ```
- [ ] Save and test

### Issue 3: Full Interference Information
- [ ] Open `Utils/fl_train.py`
- [ ] Find line 349-357: aggregation code
- [ ] Comment out mean/max/std lines
- [ ] Pass full structure to GAP edges instead
- [ ] Save and test

### Issue 5: Recompute Per Epoch
- [ ] Open `Utils/fl_train.py`
- [ ] Find `fl_train()` function (around line 104)
- [ ] Move this code inside the epoch loop:
  ```python
  # After model(batch):
  local_DS, local_PC, local_UI = component_calculate(...)
  all_DS = [local_DS] + [r['DS'] for r in clientResponse]
  ```
- [ ] Save and test

### Issue 6: Remove Circular Features
- [ ] Open `Utils/fl_train.py`
- [ ] Find line 371: `aug_batch['UE'].x = torch.cat([...])`
- [ ] Remove `global_sinr` from the concatenation
- [ ] Optionally also remove `bottleneck_indicator`
- [ ] Save and test

---

## 🔧 Debugging Commands

### Check current loss values
```python
# Add this in loss_function() after computing rate:
if torch.isnan(loss).any():
    print(f"NaN in loss!")
    print(f"Rate shape: {rate.shape}")
    print(f"Rate min/max: {rate.min()}/{rate.max()}")
    print(f"Min rate: {torch.min(rate, dim=1)[0]}")
    raise ValueError("NaN in loss")
```

### Log performance metrics
```python
# Add this in fl_eval():
print(f"Min rate: {torch.min(min_rate).item():.4f}")
print(f"Mean rate: {torch.mean(min_rate).item():.4f}")
print(f"Fairness (min/mean): {torch.min(min_rate).item() / torch.mean(min_rate).item():.2%}")
```

### Verify bottleneck type
```python
# Add this in server_return_GAP() after computing bottleneck_indicator:
print(f"Bottleneck indicator shape: {bottleneck_indicator.shape}")
print(f"Bottleneck AP indices: {bottleneck_indicator.argmax(dim=1)[:5]}")  # Sample
```

---

## ⚠️ Common Pitfalls

- ❌ **Don't change:** Model architecture without understanding information flow first
- ❌ **Don't change:** Just hyperparameters (lr, temperature) without fixing fundamental issues
- ❌ **Don't change:** Aggregation scheme without understanding what information is lost
- ✅ **Do check:** Hard-min vs soft-min - this is a likely quick win
- ✅ **Do validate:** Each fix independently before moving to the next

---

## 📈 Success Metrics

**You know it's working when:**

1. **After Issue 1 fix:** Final eval rate jumps to 1.40+ (not gradual)
2. **After Issue 2 fix:** APs start focusing on actual bottlenecked UEs
3. **After Issue 3 fix:** Convergence becomes smoother/faster
4. **After Issue 4 fix:** Training loss better correlates with eval rate
5. **After Issue 5 fix:** Fewer epochs needed for convergence

**Red flags:**
- ❌ NaN in loss → Check rate calculation for numerical issues
- ❌ Loss increases with training → Learning rate too high or loss is wrong direction
- ❌ Eval rate < 1.0 → Rate calculation is broken
- ❌ No improvement after fix → Fix wasn't applied correctly

---

## 📞 Questions to Ask When Stuck

1. **Is the fix applied?** Check with print statements before/after
2. **Is the fix in the right place?** Trace data flow: where does X come from?
3. **Is the fix logically correct?** Does it make sense mathematically?
4. **Is the fix tested?** Did you run evaluation after the change?
5. **Is there a side effect?** Did you break something else?

---

