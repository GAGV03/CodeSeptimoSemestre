import scipy.stats as st #type: ignore

print("P(Z <= <) = 0.8621. RESULTADO: " + str(st.norm.ppf(0.8621)))
print("P(Z <= z) = 0.2236. RESULTADO: " + str(st.norm.ppf(0.2236)))
print("P(-z <= Z <= z) = 0.95. RESULTADO: " + str(st.norm.ppf((1-0.95)/2)))

print("1. RESULTADO: " + str(st.norm(600,100).cdf(400)))
print("2. RESULTADO: " + str(st.norm(600,100).ppf(0.05)))