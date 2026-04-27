import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. Parametreler (Kalamar Dev Aksonu için standart değerler) ---
C_m = 1.0      # Zar kapasitansı (uF/cm^2)
g_Na = 120.0   # Maksimum Sodyum iletkenliği (mS/cm^2)
g_K = 36.0     # Maksimum Potasyum iletkenliği (mS/cm^2)
g_L = 0.3      # Sızıntı (Leak) iletkenliği (mS/cm^2)
E_Na = 50.0    # Sodyum tersine dönme potansiyeli (mV)
E_K = -77.0    # Potasyum tersine dönme potansiyeli (mV)
E_L = -54.387  # Sızıntı tersine dönme potansiyeli (mV)

# --- 2. Kapı (Gating) Değişkenleri için Hız Fonksiyonları ---
# Bu fonksiyonlar voltaja (V) bağlıdır ve iyon kanallarının açılma/kapanma hızını belirler.

def alpha_m(V):
    # m: Sodyum kanalının aktivasyon kapısı
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    # h: Sodyum kanalının inaktivasyon kapısı
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    # n: Potasyum kanalının aktivasyon kapısı
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

# --- 3. Dışarıdan Verilen Akım (Uyarım) ---
def I_inj(t):
    # 10. ile 40. milisaniyeler arasında 10 uA/cm^2'lik bir akım veriyoruz.
    if 10.0 < t < 40.0:
        return 10.0
    return 0.0

# --- 4. Diferansiyel Denklem Sistemi ---
def dALLdt(X, t):
    """
    X: [V, m, h, n] durum vektörü
    t: zaman
    """
    V, m, h, n = X
    
    # İyonik Akımların Hesaplanması
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # Membran potansiyelindeki (V) değişim
    dVdt = (I_inj(t) - I_Na - I_K - I_L) / C_m
    
    # Kapı değişkenlerindeki (m, h, n) değişim
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    
    return [dVdt, dmdt, dhdt, dndt]

# --- 5. Başlangıç Koşulları ve Simülasyon ---
# Dinlenme potansiyelini -65 mV kabul ederek kapıların başlangıç durumlarını (steady-state) hesaplıyoruz.
V0 = -65.0
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))

X0 = [V0, m0, h0, n0]

# Zaman aralığı: 0'dan 50 milisaniyeye kadar, 0.01 adımlarla
t = np.arange(0, 50, 0.01)

# Diferansiyel denklemleri çöz
X = odeint(dALLdt, X0, t)

# Sonuçları ayır
V = X[:, 0]

# --- 6. Görselleştirme ---
plt.figure(figsize=(10, 6))

# Voltaj Grafiği
plt.subplot(2, 1, 1)
plt.plot(t, V, 'b-', label='Membran Potansiyeli (V)')
plt.ylabel('Voltaj (mV)')
plt.title('Hodgkin-Huxley Aksiyon Potansiyeli Simülasyonu')
plt.grid(True)
plt.legend()

# Uyarıcı Akım Grafiği
plt.subplot(2, 1, 2)
# t dizisindeki her an için akım değerini list comprehension ile hesaplıyoruz
I_values = [I_inj(time) for time in t]
plt.plot(t, I_values, 'r-', label='Uyarıcı Akım (I)')
plt.ylabel('Akım ($\mu A/cm^2$)')
plt.xlabel('Zaman (ms)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
