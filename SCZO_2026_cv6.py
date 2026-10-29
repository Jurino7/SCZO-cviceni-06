import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.fft import fft2, fftshift
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import convolve2d
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage import exposure
from skimage.filters import sobel

plt.close('all')

# Ukol 1
# =============================================================================
# Zobrazte si jasovy profil hrany ulozene v promenne "img" na libovolnem radku
# Vytvorte 1. a 2. parcialni derivaci puvodniho obrazu a zobrazte je spolu s jasovymi profily
# =============================================================================
mat = loadmat('data/hrana.mat')
img = mat.get('a')

profile = img[1, :]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Originalni obraz')
plt.subplot(1, 2, 2)
plt.plot(profile)
plt.title('Jasovy profil')
plt.show()

first_h = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
filtered_by_1st = convolve2d(img, first_h, mode='valid')
profile_1st = filtered_by_1st[50, :]

second_h = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
filtered_by_2nd = convolve2d(img, second_h, mode='valid')
profile_2nd = filtered_by_2nd[50, :]

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Originalni obraz')
plt.subplot(2, 3, 4)
plt.plot(profile)
plt.title('Jasovy profil originalu')
plt.subplot(2, 3, 2)
plt.imshow(filtered_by_1st, cmap='gray')
plt.title('Prvni derivace')
plt.subplot(2, 3, 5)
plt.plot(profile_1st)
plt.title('Jasovy profil prvni derivace')
plt.subplot(2, 3, 3)
plt.imshow(filtered_by_2nd, cmap='gray')
plt.title('Druha derivace')
plt.subplot(2, 3, 6)
plt.plot(profile_2nd)
plt.title('Jasovy profil druhe derivace')
plt.show()


# Ukol 2
# =============================================================================
# Vyfiltrujte obraz ‘hrana.mat‘ pomoci vybraneho zostrujiciho operatoru
# Prozkoumejte funkci GetProfileWidth
# Overte funkcnost filtrace jak zobrazenim vyfiltrovanych obrazu a odpovidajicich jasovych profilu, tak vypoctenim a porovnanim sirek profilu hran
# Mente parametr K nebo alfa a pozorujte, jak se s vysledny obraz a jasove profily hrany meni
# Nactete obraz ‘MRI_brain.jpg’ a filtrujte jej pomoci Laplaceova operatoru. Zobrazte vyslednou aproximaci Laplacianu
# Zostrete obraz MRI s vyuzitim ziskaneho Laplacianu.
# =============================================================================
img = rgb2gray(img_as_float(imread('data/MRI_brain.jpg')))

mat = loadmat('data/hrana.mat')
key = [k for k in mat if not k.startswith('__')][0]
img_edge = np.asarray(mat[key], dtype=float)


def GetProfileWidth(edge_profile):
    max_intensity = np.max(edge_profile)
    bottom = max_intensity * 0.1
    top = max_intensity * 0.9
    bottom_position = (np.abs(edge_profile - bottom)).argmin()
    top_position = (np.abs(edge_profile - top)).argmin()
    profile_width = top_position - bottom_position
    return (profile_width)


L = np.ones((3, 3))
L[1, 1] = -8


def make_sharpening_operator(K):
    F = np.zeros((3, 3))
    F[1, 1] = K
    return (F + (-L)) / K  # = delta - L/K  ->  alfa = 1/K


K = 0.2
sharpening_operator = make_sharpening_operator(K)
sharpened_by_2nd = convolve2d(img_edge, sharpening_operator, mode='same', boundary='symm')

# Jasový profil = jeden řádek přes hranu (pro vodorovnou hranu použijte sloupec [:, col])
row = img_edge.shape[0] // 2
profile = img_edge[row, :]

# Overte funkcnost filtrace jak zobrazenim vyfiltrovanych obrazu a odpovidajicich jasovych profilu, tak vypoctenim a porovnanim sirek profilu hran
profile_sharpened_2nd = sharpened_by_2nd[row, :]
width_2nd = GetProfileWidth(profile_sharpened_2nd)
width_orig = GetProfileWidth(profile)

print(['Sirka originalu:', width_orig, 'Sirka po filtraci s vyuzitim druhe derivace:', width_2nd])

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img_edge, cmap='gray')
plt.title('Originalni obraz')
plt.subplot(2, 2, 3)
plt.plot(profile)
plt.title('Jasovy profil originalu')

plt.subplot(2, 2, 2)
plt.imshow(sharpened_by_2nd, cmap='gray')
plt.title('Zostreny obraz')
plt.subplot(2, 2, 4)
plt.plot(profile_sharpened_2nd)
plt.title('Jasovy profil zostreneho obrazu')
plt.show()

# Mente parametr K a pozorujte sirku hrany
for K_ in [0.1, 0.2, 0.5, 1.0, 2.0]:
    op = make_sharpening_operator(K_)
    res = convolve2d(img_edge, op, mode='same', boundary='symm')
    print(f'K = {K_}: alfa = {1 / K_:.1f}, sirka hrany = {GetProfileWidth(res[row, :])}')

# --- Část 2: MRI + Laplaceův operátor ------------------------------------------
img = rgb2gray(img_as_float(imread('data/MRI_brain.jpg')))

L = np.ones((3, 3))
L[1, 1] = -8
laplacian = convolve2d(img, L, mode='same', boundary='symm')

# Střed Laplaceova jádra je záporný -> Laplacian se ODČÍTÁ
alpha = 0.5
sharpened_img = np.clip(img - alpha * laplacian, 0, 1)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img, vmin=0, vmax=1, cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(laplacian, vmin=-1, vmax=1, cmap='gray')
plt.title('Laplacian')
plt.subplot(1, 3, 3)
plt.imshow(sharpened_img, vmin=0, vmax=1, cmap='gray')
plt.title('Zostreny Laplaceovym operatorem')
plt.show()

# Ukol 3
# =============================================================================
# Načtěte obraz ‘ugly_quidditch.jpg’
# Zobrazte si obraz spolu s jeho histogramem a spektrem
# Aplikutje na obraz vhodnou filtraci na odstranění přítomného šumu
# Na vyfiltrovaný obraz použijte vhodnou transformaci kontrastu
# Na vyfiltrovaný obraz po transformaci kontrastu použijte vhodný operátor a zvýrazněte hrany v obraze
# =============================================================================

# Načtěte obraz ‘ugly_quidditch.jpg’
img = img_as_float(imread('data/ugly_quidditch.png'))

def amplitude_spectrum(x):
    return np.log(1 + fftshift(np.abs(fft2(x))))


def show_with_analysis(x, title, fig_num):
    """Obraz + histogram + amplitudové spektrum."""
    plt.figure(fig_num, figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(x, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.subplot(1, 3, 2)
    plt.hist(x.ravel(), bins=256, range=(0, 1), color='k')
    plt.title('Histogram')
    plt.xlabel('Jasova hodnota')
    plt.ylabel('Cetnost')
    plt.subplot(1, 3, 3)
    M, N = x.shape
    plt.imshow(amplitude_spectrum(x), cmap='gray', extent=[-N / 2, N / 2, M / 2, -M / 2])
    plt.title('Amplitudove spektrum')
    plt.xlabel('Prostorova frekvence [1/m]')
    plt.ylabel('Prostorova frekvence [1/m]')
    plt.tight_layout()


# Zobrazte si obraz spolu s jeho histogramem a spektrem
show_with_analysis(img, 'Puvodni obraz', 'Ukol 3 - analyza')
plt.show()

# -----------------------------------------------------------------------------
# Aplikujte na obraz vhodnou filtraci na odstranění přítomného šumu
# Volba podle toho, co je vidět v obraze / histogramu / spektru:
#   'median'   - impulsní šum (sůl a pepř): v histogramu špičky na 0 a 1
#   'gaussian' - gaussovský/bílý šum: ve spektru plochá vysokofrekvenční "podlaha"
# -----------------------------------------------------------------------------
noise_filter = 'median'

if noise_filter == 'median':
    denoised = median_filter(img, size=3)
else:
    denoised = gaussian_filter(img, sigma=1.0)

# -----------------------------------------------------------------------------
# Na vyfiltrovaný obraz použijte vhodnou transformaci kontrastu
#   'stretch'  - lineární roztažení histogramu (úzký histogram)
#   'equalize' - vyrovnání histogramu (nerovnoměrné rozložení jasů)
#   'gamma'    - gama korekce (celkově příliš tmavý / světlý obraz)
# -----------------------------------------------------------------------------
contrast = 'stretch'

if contrast == 'stretch':
    p_low, p_high = np.percentile(denoised, (2, 98))
    contrasted = exposure.rescale_intensity(denoised, in_range=(p_low, p_high), out_range=(0, 1))
elif contrast == 'equalize':
    contrasted = exposure.equalize_hist(denoised)
else:
    contrasted = exposure.adjust_gamma(denoised, gamma=0.6)  # gamma < 1 zesvětlí, > 1 ztmaví

# -----------------------------------------------------------------------------
# Na vyfiltrovaný obraz po transformaci kontrastu použijte vhodný operátor
# a zvýrazněte hrany v obraze
# -----------------------------------------------------------------------------
L = np.ones((3, 3))
L[1, 1] = -8
laplacian = convolve2d(contrasted, L, mode='same', boundary='symm')
alpha = 0.5
sharpened = np.clip(contrasted - alpha * laplacian, 0, 1)  # střed jádra záporný -> odečítáme

edges = sobel(contrasted)  # velikost gradientu (samotné hrany)

show_with_analysis(denoised, 'Po odstraneni sumu', 'Ukol 3 - po filtraci')
show_with_analysis(contrasted, 'Po transformaci kontrastu', 'Ukol 3 - po kontrastu')

plt.figure('Ukol 3 - vysledek', figsize=(16, 4.5))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Puvodni')
plt.subplot(1, 4, 2)
plt.imshow(denoised, cmap='gray', vmin=0, vmax=1)
plt.title(f'Bez sumu ({noise_filter})')
plt.subplot(1, 4, 3)
plt.imshow(sharpened, cmap='gray', vmin=0, vmax=1)
plt.title(f'Kontrast ({contrast}) + zostreni hran')
plt.subplot(1, 4, 4)
plt.imshow(edges, cmap='gray')
plt.title('Hrany (Sobel)')
for a in plt.gcf().axes:
    a.axis('off')
plt.tight_layout()
plt.show()

