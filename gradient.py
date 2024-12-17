x = [1, 2, 3]  
y = [2, 4, 6]  

alpha = 0.1  
iterations = 5  

m = 0
c = 0

def compute_gradients(x, y, m, c):
    n = len(x)
    dm = 0  
    dc = 0  
    
    for i in range(n):
        y_pred = m * x[i] + c
        error = y[i] - y_pred
        dm += -2 * x[i] * error  
        dc += -2 * error         
    
    dm /= n
    dc /= n
    return dm, dc

print("Iteration\tm\t\tc")
for i in range(iterations):
    dm, dc = compute_gradients(x, y, m, c)
    m -= alpha * dm  
    c -= alpha * dc  
    print(f"{i+1}\t\t{m:.4f}\t\t{c:.4f}")

print("\nFinal values:")
print(f"m = {m:.4f}")
print(f"c = {c:.4f}")
