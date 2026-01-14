import numpy as np

# Setup
N = 100
base_val = np.float32(100_000.0)  # Range: 10^5
correction = np.float32(0.4)  # We want to shift the mean by 0.08

# 1. Accumulator Approach (Sum then Divide)
# The sum will grow to 100,000,000 (10^8)
acc_sum = np.float32(0.0)
for _ in range(N):
    acc_sum += np.float32(base_val)

# AT THIS POINT: acc_sum is 100,000,000.
# In float32, the gap between numbers at 10^8 is 8.0!
# Any value smaller than 4.0 added to this sum is mathematically deleted.

acc_sum += np.float32(correction)
sum_mean = acc_sum / (N + 1)

# 2. Incremental Averaging Approach
# The running average stays near 100,000 (10^5)
inc_avg = np.float32(0.0)
for i in range(N):
    inc_avg += (np.float32(base_val) - inc_avg) / (i + 1)

# AT THIS POINT: inc_avg is 100,000.
# In float32, the gap between numbers at 10^5 is only 0.0078.
# We can resolve updates 1000x smaller than the sum-based approach can.

inc_avg += (np.float32(correction) - inc_avg) / (N + 1)

# 3. Array, then mean.
array_vals = np.full((N + 1,), np.float32(base_val), dtype=np.float32)
array_vals[-1] = np.float32(correction)
array_mean = np.mean(array_vals)

# Ground Truth (float64)
true_mean = (float(base_val) * N + float(correction)) / (N + 1)

print(f"True Mean: {true_mean:.6f}")
print(f"Sum-based Mean: {sum_mean:.6f}, error: {abs(true_mean - float(sum_mean)):.2e}")
print(f"Incremental Mean: {inc_avg:.6f}, error: {abs(true_mean - float(inc_avg)):.2e}")
print(f"Aray Mean: {array_mean:.6f}, error: {abs(true_mean - float(array_mean)):.2e}")
